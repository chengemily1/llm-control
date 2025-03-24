from src.steering_methods.control_wrapper import LiSeCoWrapper, MultiDimLiSeCoWrapper

def print_intervention_summary(model_name: str, method: str, selected_layers: list, 
                             liseco_params: dict = None) -> None:
    """Print a summary of the intervention configuration.
    
    Args:
        model_name (str): Name of the model
        method (str): Intervention method ('liseco', 'baseline', or 'instruct')
        selected_layers (list): List of layers to be controlled
        liseco_params (dict, optional): Parameters for LiSeCo method, containing:
            - liseco_lower: Lower bound
            - liseco_upper: Upper bound
            - liseco_map: Mapping function
            - s: Scale factor (optional)
    """
    print("\n" + "="*50)
    print("INTERVENTION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Method: {method}")
    
    if method == 'liseco' and liseco_params:
        print(f"Layers to be controlled: {sorted(selected_layers)}")
        print(f"Control parameters:")
        print(f"  - Lower bound: {liseco_params['liseco_lower']}")
        print(f"  - Upper bound: {liseco_params['liseco_upper']}")
        print(f"  - Mapping function: {liseco_params['liseco_map']}")
        if liseco_params.get('s') is not None:
            print(f"  - Scale factor: {liseco_params['s']}")
    else:
        print("No layer intervention (baseline or instruct method)")
    print("="*50 + "\n")

def retrofit_model(model, layerlist, Ws, args):
    """Wrap all layers of the model with appropriate wrappers."""
    num_layers = model.config.num_hidden_layers
    for layer in range(num_layers):
        if type(layerlist[layer]) != LiSeCoWrapper:
            layerlist[layer] = LiSeCoWrapper(
                layerlist[layer],
                linear_probe=Ws[layer],
                lower=args.liseco_lower,
                upper=args.liseco_upper,
                map_to_target_space=args.liseco_map
            )
        else:
            layerlist[layer] = LiSeCoWrapper(
                layerlist[layer].base_layer,
                linear_probe=Ws[layer],
                lower=args.liseco_lower,
                upper=args.liseco_upper,
                map_to_target_space=args.liseco_map
            )

def retrofit_model_multidim(model, layerlist, Ws, args):
    """Wrap all layers of the model with appropriate wrappers."""
    num_layers = model.config.num_hidden_layers
    for layer in range(num_layers):
        if type(layerlist[layer]) != MultiDimLiSeCoWrapper:
            layerlist[layer] = MultiDimLiSeCoWrapper(
                layerlist[layer],
                linear_probe=Ws[layer],
                device=args.device
            )
        else:
            layerlist[layer] = MultiDimLiSeCoWrapper(
                layerlist[layer].base_layer,
                linear_probe=Ws[layer],
                device=args.device
            )

def collect_layer_metrics(layerlist, layer):
    """Collect metrics from a specific layer."""
    return {
        'pre_adjust_toxicity_prob': [float(score.item()) for score in layerlist[layer].pre_adjust_toxicity_log.copy()],
        'post_adjust_toxicity_prob': [float(score.item()) for score in layerlist[layer].post_adjust_toxicity_log.copy()],
        'inference_latency': layerlist[layer].latency.copy()
    } 