"""Control styles copied from paper/plot_style.py; attribution in provenance."""
COLORS = dict(baseline1_hdv='#B8BDC7', baseline2_matched='#CC79A7', noop='#6B7280', left_if_safe='#8C6D31')
MARKERS = dict(baseline1_hdv='o', baseline2_matched='D', noop='x', left_if_safe='v')
LINESTYLES = dict(baseline1_hdv=':', baseline2_matched=(0, (6, 2)), noop=(0, (3, 1, 1, 1)), left_if_safe=(0, (5, 2, 1, 2)))
MARKER_FACE_COLORS = {key: 'white' if key == 'baseline1_hdv' else value for key, value in COLORS.items()}
