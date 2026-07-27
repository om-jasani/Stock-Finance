"""Color schemes and style constants"""

class ColorScheme:
    """Application color scheme"""
    
    # Main colors
    PRIMARY = "#0078D4"
    SECONDARY = "#404040"
    BACKGROUND = "#1E1E1E"
    SURFACE = "#2D2D2D"
    ERROR = "#FF5252"
    SUCCESS = "#4CAF50"
    WARNING = "#FFC107"
    INFO = "#2196F3"
    
    # Text colors
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#B3B3B3"
    TEXT_DISABLED = "#666666"
    
    # Chart colors
    CHART_UP = "#00C805"
    CHART_DOWN = "#FF5252"
    CHART_VOLUME_UP = "#265828"
    CHART_VOLUME_DOWN = "#642424"
    
    # Technical indicators
    INDICATOR_SMA_20 = "#FFA726"
    INDICATOR_SMA_50 = "#29B6F6"
    INDICATOR_BB_UPPER = "#B388FF"
    INDICATOR_BB_LOWER = "#B388FF"
    INDICATOR_BB_MIDDLE = "#7C4DFF"
    
    # Gradient colors
    GRADIENT_START = "#1A237E"
    GRADIENT_END = "#0D47A1"
    
    @classmethod
    def get_sequential_colors(cls, n: int) -> list:
        """Get n distinct colors for sequential data"""
        base_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]
        return base_colors[:n]
        
    @classmethod
    def get_diverging_colors(cls, n: int) -> list:
        """Get n distinct colors for diverging data"""
        base_colors = [
            "#d73027", "#f46d43", "#fdae61", "#fee090", "#ffffbf",
            "#e0f3f8", "#abd9e9", "#74add1", "#4575b4"
        ]
        return base_colors[:n]

class StyleConstants:
    """Style constants for the application"""
    
    # Font settings
    FONT_FAMILY = "Segoe UI"
    FONT_SIZE_SMALL = 11
    FONT_SIZE_NORMAL = 12
    FONT_SIZE_LARGE = 14
    FONT_SIZE_HEADER = 18
    
    # Spacing
    PADDING_SMALL = 5
    PADDING_NORMAL = 10
    PADDING_LARGE = 15
    MARGIN_SMALL = 5
    MARGIN_NORMAL = 10
    MARGIN_LARGE = 15
    
    # Border radius
    BORDER_RADIUS_SMALL = 4
    BORDER_RADIUS_NORMAL = 6
    BORDER_RADIUS_LARGE = 8
    
    # Animation durations (ms)
    ANIMATION_FAST = 150
    ANIMATION_NORMAL = 250
    ANIMATION_SLOW = 350
    
    # Chart settings
    CHART_HEIGHT_SMALL = 300
    CHART_HEIGHT_NORMAL = 400
    CHART_HEIGHT_LARGE = 500
    CANDLESTICK_WIDTH = 0.8
    
    # Widget sizes
    BUTTON_HEIGHT = 32
    INPUT_HEIGHT = 32
    HEADER_HEIGHT = 50
    SIDEBAR_WIDTH = 250
    
    # Shadow settings
    SHADOW_COLOR = "rgba(0, 0, 0, 0.2)"
    SHADOW_OFFSET = "2px 2px"
    SHADOW_BLUR = "4px"
    
    @classmethod
    def get_shadow(cls, level: int = 1) -> str:
        """Get shadow style based on elevation level"""
        if level == 0:
            return "none"
        elif level == 1:
            return f"{cls.SHADOW_OFFSET} {cls.SHADOW_BLUR} {cls.SHADOW_COLOR}"
        elif level == 2:
            return f"{cls.SHADOW_OFFSET} {cls.SHADOW_BLUR} {cls.SHADOW_COLOR}, " \
                   f"4px 4px 8px {cls.SHADOW_COLOR}"
        else:
            return f"{cls.SHADOW_OFFSET} {cls.SHADOW_BLUR} {cls.SHADOW_COLOR}, " \
                   f"4px 4px 8px {cls.SHADOW_COLOR}, " \
                   f"8px 8px 16px {cls.SHADOW_COLOR}"