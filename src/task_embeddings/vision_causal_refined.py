"""Causal checks for cold-query parameter-group retrieval in a pretrained ViT."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import save_json, seed_everything
from .domain_optimizer import ParameterPartition
from .qwen_sensitivity import cold_fold_predictions


def select_parameter_budget(
    scores: torch.Tensor,
    sizes: torch.Tensor,
    scope: torch.Tensor,
    fraction: float,
) -> tuple[torch.Tensor, float]:
    """Select highest-scoring groups until a scoped parameter budget is met."""
    if scores.ndim != 1 or sizes.shape != scores.shape or scope.shape != scores.shape:
        raise ValueError("scores, sizes, and scope must be aligned vectors")
    if not 0 < fraction < 1 or not bool(scope.any()):
        raise ValueError("fraction must be in (0, 1) and scope must be nonempty")
    scores = scores.detach().double().cpu()
    sizes = sizes.detach().double().cpu()
    scope = scope.detach().bool().cpu()
    eligible = scope.nonzero().flatten()
    order = eligible[torch.argsort(scores[eligible], descending=True, stable=True)]
    target = fraction * sizes[scope].sum()
    cumulative = sizes[order].cumsum(0)
    count = int((cumulative < target).sum()) + 1
    selected = torch.zeros_like(scope)
    selected[order[:count]] = True
    actual = float(sizes[selected].sum() / sizes[scope].sum())
    return selected, actual


@contextmanager
def mean_ablate_groups(partition: ParameterPartition, selected: torch.Tensor):
    """Replace selected rows/columns by their within-tensor mean, then restore."""
    if selected.shape != (partition.n_groups,):
        raise ValueError("selection must contain one flag per parameter group")
    backups = []
    try:
        with torch.no_grad():
            for spec in partition.slices:
                local = selected[spec.offset : spec.offset + spec.count]
                indices = local.nonzero().flatten().to(spec.parameter.device)
                if not len(indices):
                    continue
                if spec.axis is None:
                    backups.append((spec, None, spec.parameter.detach().clone()))
                    spec.parameter.fill_(spec.parameter.mean())
                    continue
                original = spec.parameter.index_select(spec.axis, indices).clone()
                backups.append((spec, indices, original))
                replacement = spec.parameter.mean(dim=spec.axis, keepdim=True)
                shape = list(spec.parameter.shape)
                shape[spec.axis] = len(indices)
                spec.parameter.index_copy_(
                    spec.axis, indices, replacement.expand(shape)
                )
        yield
    finally:
        with torch.no_grad():
            for spec, indices, original in reversed(backups):
                if spec.axis is None:
                    spec.parameter.copy_(original)
                else:
                    spec.parameter.index_copy_(spec.axis, indices, original)


def transformer_block_scope(partition: ParameterPartition) -> torch.Tensor:
    scope = torch.zeros(partition.n_groups, dtype=torch.bool)
    for spec in partition.slices:
        if spec.name.startswith("blocks."):
            scope[spec.offset : spec.offset + spec.count] = True
    return scope


def mlp_feature_layout(model, partition: ParameterPartition):
    """Return coupled ViT MLP feature ranges and their fc2 input modules."""
    modules = dict(model.named_modules())
    layout = []
    for spec in partition.slices:
        if spec.name.endswith(".mlp.fc1.weight"):
            prefix = spec.name.removesuffix(".fc1.weight")
            output_name = prefix + ".fc2"
        elif spec.name.endswith(".mlp.gate_proj.weight"):
            prefix = spec.name.removesuffix(".gate_proj.weight")
            output_name = prefix + ".down_proj"
        else:
            continue
        layout.append(
            {
                "name": prefix,
                "offset": spec.offset,
                "count": spec.count,
                "module": modules[output_name],
            }
        )
    if not layout:
        raise ValueError("model has no coupled ViT MLP features")
    return layout


def mlp_feature_scope(partition: ParameterPartition, layout) -> torch.Tensor:
    scope = torch.zeros(partition.n_groups, dtype=torch.bool)
    for item in layout:
        scope[item["offset"] : item["offset"] + item["count"]] = True
    return scope


@torch.no_grad()
def calibrate_activation_means(model, dataset, indices: torch.Tensor, layout, batch_size=16):
    """Estimate disjoint-reference means at the inputs to each MLP fc2."""
    totals = {
        item["name"]: torch.zeros(item["count"], device=next(model.parameters()).device)
        for item in layout
    }
    counts = {item["name"]: 0 for item in layout}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            activation = inputs[0].detach().float()
            dims = tuple(range(activation.ndim - 1))
            totals[name].add_(activation.sum(dim=dims))
            counts[name] += activation.numel() // activation.shape[-1]

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        for local in indices.split(batch_size):
            images = torch.stack([dataset[int(index)][0] for index in local])
            model(images.to(next(model.parameters()).device))
    finally:
        for handle in handles:
            handle.remove()
    return {name: value / counts[name] for name, value in totals.items()}


@contextmanager
def mean_ablate_mlp_activations(layout, selected: torch.Tensor, means):
    """Replace selected coupled MLP feature activations by calibration means."""
    handles = []
    for item in layout:
        local = selected[item["offset"] : item["offset"] + item["count"]]
        indices = local.nonzero().flatten()
        if not len(indices):
            continue
        name = item["name"]

        def intervene(_module, inputs, indices=indices, name=name):
            activation = inputs[0].clone()
            device_indices = indices.to(activation.device)
            replacement = means[name][device_indices].to(activation.dtype)
            activation[..., device_indices] = replacement
            return (activation, *inputs[1:])

        handles.append(item["module"].register_forward_pre_hook(intervene))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def zero_ablate_mlp_activations(layout, selected: torch.Tensor):
    """Set selected coupled MLP feature activations to zero."""
    handles = []
    for item in layout:
        local = selected[item["offset"] : item["offset"] + item["count"]]
        indices = local.nonzero().flatten()
        if not len(indices):
            continue

        def intervene(_module, inputs, indices=indices):
            activation = inputs[0].clone()
            activation[..., indices.to(activation.device)] = 0
            return (activation, *inputs[1:])

        handles.append(item["module"].register_forward_pre_hook(intervene))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def activation_feature_scores(model, image, label: int, functional: str, partition, layout):
    """Query-specific activation and activation-times-gradient feature scores."""
    captured = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            captured[name] = inputs[0]
            captured[name].retain_grad()

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(next(model.parameters()).device)).float()
        if functional == "class_logit":
            objective = logits[0, label]
        elif functional == "margin":
            alternatives = logits[0].clone()
            alternatives[label] = -torch.inf
            objective = logits[0, label] - alternatives.max()
        elif functional == "loss":
            objective = F.cross_entropy(
                logits, torch.tensor([label], device=logits.device)
            )
        else:
            raise ValueError(functional)
        objective.backward()
        magnitude = torch.zeros(partition.n_groups, dtype=torch.float64)
        act_grad = torch.zeros_like(magnitude)
        for item in layout:
            value = captured[item["name"]]
            dims = tuple(range(value.ndim - 1))
            local_magnitude = value.detach().float().square().mean(dim=dims)
            local_act_grad = (
                value.detach().float() * value.grad.detach().float()
            ).square().mean(dim=dims)
            sl = slice(item["offset"], item["offset"] + item["count"])
            magnitude[sl] = local_magnitude.double().cpu()
            act_grad[sl] = local_act_grad.double().cpu()
        return magnitude, act_grad
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def mean_ablation_taylor(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    mean: torch.Tensor,
) -> torch.Tensor:
    """First-order functional degradation from replacing features by means."""
    if activation.shape != gradient.shape or activation.shape[-1] != len(mean):
        raise ValueError("activation, gradient, and feature mean must align")
    dims = tuple(range(activation.ndim - 1))
    return (gradient * (activation - mean)).sum(dim=dims)


def mean_ablation_taylor_scores(
    model,
    image,
    label: int,
    functional: str,
    partition,
    layout,
    means,
):
    """Coordinate-matched Taylor scores for the calibrated mean intervention."""
    captured = {}
    handles = []
    for item in layout:
        name = item["name"]

        def capture(_module, inputs, name=name):
            captured[name] = inputs[0]
            captured[name].retain_grad()

        handles.append(item["module"].register_forward_pre_hook(capture))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image[None].to(next(model.parameters()).device)).float()
        if functional == "class_logit":
            objective = logits[0, label]
            direction = 1
        elif functional == "margin":
            alternatives = logits[0].clone()
            alternatives[label] = -torch.inf
            objective = logits[0, label] - alternatives.max()
            direction = 1
        elif functional == "loss":
            objective = F.cross_entropy(
                logits, torch.tensor([label], device=logits.device)
            )
            direction = -1
        else:
            raise ValueError(functional)
        objective.backward()
        scores = torch.zeros(partition.n_groups, dtype=torch.float64)
        for item in layout:
            value = captured[item["name"]].detach().float()
            gradient = captured[item["name"]].grad.detach().float()
            mean = means[item["name"]].to(value.device).float()
            local = mean_ablation_taylor(value, gradient, mean) * direction
            sl = slice(item["offset"], item["offset"] + item["count"])
            scores[sl] = local.double().cpu()
        return scores
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def functional_value(model, image: torch.Tensor, label: int, functional: str) -> float:
    logits = model(image[None].to(next(model.parameters()).device)).float()
    if functional == "class_logit":
        value = logits[0, label]
    elif functional == "margin":
        alternatives = logits[0].clone()
        alternatives[label] = -torch.inf
        value = logits[0, label] - alternatives.max()
    elif functional == "loss":
        value = F.cross_entropy(logits, torch.tensor([label], device=logits.device))
    else:
        raise ValueError(functional)
    return float(value)


def run_causal_study(
    model,
    dataset,
    payload: dict[str, object],
    *,
    queries: int = 24,
    folds: int = 4,
    fractions: tuple[float, ...] = (0.0002, 0.001),
    seed: int = 23_041,
) -> dict[str, object]:
    """Mean-ablate retrieved transformer-block groups on held-out query images."""
    partition = ParameterPartition(model, "swiglu")
    scope = transformer_block_scope(partition)
    sizes = partition.sizes.detach().double().cpu()
    target_indices = payload["target_indices"].long()
    class_indices = payload["selected_class_indices"].long()
    feature_maps = payload["feature_maps"]
    features = feature_maps["frozen_input_encoder"].double()
    query_columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    weight_magnitude = partition.weight_energy().detach().double().cpu() * sizes
    records = []
    generator = torch.Generator().manual_seed(seed)

    for functional in ("class_logit", "margin", "loss"):
        source = payload[functional]["source_profiles"].double()
        target = payload[functional]["target_profiles"].double()
        cold = cold_fold_predictions(source, features, seed=seed, folds=folds)
        score_grid = {
            "dinov2_affine": cold["affine_semantic"],
            "dinov2_rbf": cold["rbf_semantic"],
            "scalar_mass": cold["scalar_mass"],
            "nearest": cold["nearest"],
            "source_onehot_reference": source,
            "direct_gradient_oracle": target,
            "weight_magnitude": weight_magnitude[:, None].expand_as(target),
        }
        for replicate in range(3):
            score_grid[f"random_{replicate}"] = torch.rand(
                target.shape, generator=generator, dtype=torch.float64
            )
        del cold

        for column in query_columns.tolist():
            image, label = dataset[int(target_indices[column])]
            label = int(label)
            if label != int(class_indices[column]):
                raise RuntimeError("stored class order does not match the dataset")
            baseline = functional_value(model, image, label, functional)
            with torch.no_grad():
                predicted = int(model(image[None].to(partition.device)).argmax(1))
            for method, scores in score_grid.items():
                for fraction in fractions:
                    selected, actual = select_parameter_budget(
                        scores[:, column].clamp_min(0), sizes, scope, fraction
                    )
                    with mean_ablate_groups(partition, selected):
                        intervened = functional_value(
                            model, image, label, functional
                        )
                    effect = (
                        intervened - baseline
                        if functional == "loss"
                        else baseline - intervened
                    )
                    records.append(
                        {
                            "functional": functional,
                            "method": method,
                            "query_column": column,
                            "class_index": label,
                            "target_correct": predicted == label,
                            "requested_parameter_fraction": fraction,
                            "actual_scoped_parameter_fraction": actual,
                            "selected_groups": int(selected.sum()),
                            "selected_parameters": int(sizes[selected].sum()),
                            "baseline": baseline,
                            "intervened": intervened,
                            "degradation": effect,
                        }
                    )
    return {
        "protocol": {
            "intervention": "within-tensor mean ablation",
            "scope": "transformer blocks only; patch embedding, final norm, and classifier head excluded",
            "queries": len(query_columns),
            "query_columns": query_columns.tolist(),
            "fractions": list(fractions),
            "random_replicates": 3,
            "cold_representation": "facebook/dinov2-small",
        },
        "resources": {
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
            "parameters": partition.n_parameters,
            "scoped_parameters": int(sizes[scope].sum()),
        },
        "records": records,
    }


def run_activation_causal_study(
    model,
    dataset,
    payload: dict[str, object],
    *,
    queries: int = 24,
    folds: int = 4,
    fractions: tuple[float, ...] = (0.0002, 0.001),
    functionals: tuple[str, ...] = ("margin",),
    calibration_images: int = 64,
    seed: int = 23_041,
) -> dict[str, object]:
    """Mean-ablate MLP activations selected by parameter sensitivity profiles."""
    partition = ParameterPartition(model, "swiglu")
    layout = mlp_feature_layout(model, partition)
    scope = mlp_feature_scope(partition, layout)
    sizes = partition.sizes.detach().double().cpu()
    target_indices = payload["target_indices"].long()
    source_indices = payload["source_indices"].long()
    class_indices = payload["selected_class_indices"].long()
    features = payload["feature_maps"]["frozen_input_encoder"].double()
    reference_indices = payload.get("representation_indices")
    if reference_indices is None:
        # Older refined artifacts predate this metadata field. Their fixed
        # protocol stored four representation images immediately before the
        # source image within each ImageFolder class.
        reference_indices = torch.cat(
            [torch.arange(int(index) - 4, int(index)) for index in source_indices]
        )
    positions = torch.linspace(
        0, len(reference_indices) - 1, min(calibration_images, len(reference_indices))
    ).round().long().unique()
    means = calibrate_activation_means(
        model, dataset, reference_indices[positions], layout
    )
    query_columns = torch.linspace(
        0, len(target_indices) - 1, min(queries, len(target_indices))
    ).round().long().unique()
    weight_magnitude = partition.weight_energy().detach().double().cpu() * sizes
    generator = torch.Generator().manual_seed(seed)
    records = []

    for functional in functionals:
        source = payload[functional]["source_profiles"].double()
        target = payload[functional]["target_profiles"].double()
        cold = cold_fold_predictions(source, features, seed=seed, folds=folds)
        static_scores = {
            "dinov2_affine": cold["affine_semantic"],
            "dinov2_rbf": cold["rbf_semantic"],
            "scalar_mass": cold["scalar_mass"],
            "nearest": cold["nearest"],
            "source_onehot_reference": source,
            "direct_gradient_oracle": target,
            "weight_magnitude": weight_magnitude[:, None].expand_as(target),
            "random": torch.rand(target.shape, generator=generator, dtype=torch.float64),
        }
        for column in query_columns.tolist():
            target_image, label = dataset[int(target_indices[column])]
            source_image, source_label = dataset[int(source_indices[column])]
            label = int(label)
            if label != int(class_indices[column]) or int(source_label) != label:
                raise RuntimeError("stored class order does not match the dataset")
            source_magnitude, source_act_grad = activation_feature_scores(
                model, source_image, label, functional, partition, layout
            )
            target_magnitude, target_act_grad = activation_feature_scores(
                model, target_image, label, functional, partition, layout
            )
            score_grid = {
                **{name: values[:, column] for name, values in static_scores.items()},
                "source_activation_magnitude": source_magnitude,
                "source_activation_x_gradient": source_act_grad,
                "target_activation_magnitude": target_magnitude,
                "target_activation_x_gradient": target_act_grad,
            }
            baseline = functional_value(model, target_image, label, functional)
            with torch.no_grad():
                predicted = int(
                    model(target_image[None].to(partition.device)).argmax(1)
                )
            for method, scores in score_grid.items():
                for fraction in fractions:
                    selected, actual = select_parameter_budget(
                        scores.clamp_min(0), sizes, scope, fraction
                    )
                    with mean_ablate_mlp_activations(layout, selected, means):
                        intervened = functional_value(
                            model, target_image, label, functional
                        )
                    effect = (
                        intervened - baseline
                        if functional == "loss"
                        else baseline - intervened
                    )
                    records.append(
                        {
                            "functional": functional,
                            "method": method,
                            "query_column": column,
                            "class_index": label,
                            "target_correct": predicted == label,
                            "requested_parameter_fraction": fraction,
                            "actual_scoped_parameter_fraction": actual,
                            "selected_groups": int(selected.sum()),
                            "selected_parameters": int(sizes[selected].sum()),
                            "baseline": baseline,
                            "intervened": intervened,
                            "degradation": effect,
                        }
                    )
    return {
        "protocol": {
            "intervention": "disjoint-calibration mean ablation at MLP fc2 inputs",
            "scope": "coupled transformer-block MLP activation features",
            "queries": len(query_columns),
            "query_columns": query_columns.tolist(),
            "fractions": list(fractions),
            "functionals": list(functionals),
            "calibration_images": len(positions),
            "cold_representation": "facebook/dinov2-small",
        },
        "resources": {
            "groups": partition.n_groups,
            "scoped_groups": int(scope.sum()),
            "parameters": partition.n_parameters,
            "scoped_parameters": int(sizes[scope].sum()),
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="vit_base_patch16_224.augreg2_in21k_ft_in1k"
    )
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--activation-aligned", action="store_true")
    parser.add_argument("--calibration-images", type=int, default=64)
    args = parser.parse_args()
    seed_everything(23_041)

    import timm
    from torchvision.datasets import ImageFolder

    model = timm.create_model(args.model, pretrained=True).cuda().eval()
    transform = timm.data.create_transform(
        **timm.data.resolve_model_data_config(model), is_training=False
    )
    dataset = ImageFolder(args.data / "validation", transform=transform)
    payload = torch.load(args.profiles, map_location="cpu", weights_only=True)
    if args.activation_aligned:
        result = run_activation_causal_study(
            model,
            dataset,
            payload,
            queries=args.queries,
            folds=args.folds,
            calibration_images=args.calibration_images,
        )
    else:
        result = run_causal_study(
            model, dataset, payload, queries=args.queries, folds=args.folds
        )
    result["model"] = args.model
    result["profiles"] = str(args.profiles)
    save_json(args.output, result)


if __name__ == "__main__":
    main()
