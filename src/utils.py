# -*- coding:utf-8 -*-
__projet__ = "GeoclassificationMPS"
__nom_fichier__ = "utils"
__author__ = "MENGELLE Axel"
__date__ = "sept 2024"


def cartesian_product(*arrays):
    # Base case: if no arrays are given, return an empty tuple
    if not arrays:
        return [()]
    
    # Recursive case: process the first array and combine with the rest
    first_array, *rest_arrays = arrays
    rest_product = cartesian_product(*rest_arrays)
    
    # Combine elements from the first array with the result of the recursive call
    result = []
    for element in first_array:
        for combination in rest_product:
            result.append((element,) + combination)
    
    return result
