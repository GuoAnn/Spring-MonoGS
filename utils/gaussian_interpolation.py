import numpy as np

def interpolate_gaussian(g1, g2, alpha):
    """
    对两个高斯点g1, g2按alpha进行插值，返回新高斯点
    g1, g2: dict或自定义高斯点对象
    alpha: float, 0~1
    """
    # 位置插值
    position = (1 - alpha) * g1['position'] + alpha * g2['position']
    # 颜色插值
    color = (1 - alpha) * g1['color'] + alpha * g2['color']
    # 不透明度插值
    opacity = (1 - alpha) * g1['opacity'] + alpha * g2['opacity']
    # 协方差插值
    scale = (1 - alpha) * g1['scale'] + alpha * g2['scale']
    # 旋转插值（如有）
    # rotation = slerp(g1['rotation'], g2['rotation'], alpha)
    # 其他属性同理
    return {
        'position': position,
        'color': color,
        'opacity': opacity,
        'scale': scale,
        # 'rotation': rotation,
        # ...
    }