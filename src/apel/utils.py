from pycram.datastructures.dataclasses import Color


def hex_to_rgba(hex_color, alpha=1.0) -> Color:
    hex_color = hex_color.lstrip('#')
    rgba = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)) + (alpha,)
    return Color(*rgba)
