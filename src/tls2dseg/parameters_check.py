

def check_all_parameters(task_parameters, pcp_parameters, image_generation_parameters,
                         inference_models_parameters, slice_inference_parameters, text_prompt) -> None:
    # Check all input parameters if they adhere to expected values and types

    # pcp_parameters
    # __________________________________________________________________________________________________________________
    # Range limits
    range_limits = pcp_parameters['range_limits']
    if len(range_limits) != 2:
        raise ValueError(f"Range limits provided, but not specifying both min and max range!")
    if not all(isinstance(x, (int, float)) for x in range_limits):
        raise TypeError("Both elements of range_limits must be numbers (int or float)!")

    # Region of Interest (RoI)
    roi_limits = pcp_parameters['roi_limits']
    if len(roi_limits) != 6:
        raise ValueError(f"RoI limits provided, but len() != 6, not specifying min and max of X,Y and Z!")
    if not all(isinstance(x, (int, float)) for x in roi_limits):
        raise TypeError("All elements of roi_limits must be numbers (int or float)!")

    return None
