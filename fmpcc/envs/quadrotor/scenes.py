import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model', 'scenes')
SCENES = {
    'corridor': 'scene_corridor.xml',
    'corridor_wide': 'scene_corridor_v2.xml',
    's_curve': 'scene_s_curve.xml',
    'pillars': 'scene_avoiding_pillars_s36.xml',
}
LANES = {'corridor': ['L', 'C', 'R'], 's_curve': ['default']}
CONTACT_LIMIT = {'corridor': 0.02, 's_curve': 0.08}
MIN_ALTITUDE = 0.50


def xml(scene):
    return os.path.join(MODEL_DIR, SCENES[scene])


def obstacle_contact(model, contact):
    return model.geom(contact.geom1).name != 'floor' and model.geom(contact.geom2).name != 'floor'
