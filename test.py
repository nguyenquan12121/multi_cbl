import Sofa
import time
import threading
import numpy as np
import math

class BallMovementController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.mechanical_object = kwargs.get("mechanical_object")
        self.speed = 10.0
        print("Ball Movement Controller initialized")
        print(self.mechanical_object)

    def onKeypressedEvent(self, event):
        key = event['key']
        if ord(key) in [19, 21, 18, 20]:  # ASCII codes for arrow keys
            random_translation = np.random.uniform(-100, 100, 3).tolist()
            random_rotation = [0., 0., 0.]
            random_position = [random_translation + random_rotation + [1.0]]
            self.mechanical_object.position.value = random_position
            print(f"New random position: {random_position}")

    def onScriptEvent(self, event):
        if 'integer_value' in event:
            value = event['integer_value']
            self.mechanical_object.position.value = value
            print(f"<Receiver> Received and stored list: {value}")

class IntegerGeneratorController(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self, *args, **kwargs)
        self.counter = 0
        self.running = False
        self.receiver = kwargs.get("receiverController")
        print("Sender initialized with receiver:", self.receiver)
        self.timer_thread = None

    def init(self):
        self.running = True
        self.timer_thread = threading.Thread(target=self.timer_function)
        self.timer_thread.daemon = True
        self.timer_thread.start()

    def timer_function(self):
        # Send int every second
        circle_size = 100
        while self.running:
            time.sleep(1/30)
            
            curr_time = time.time() % 100
            x = circle_size * math.sin(curr_time)
            y = circle_size *  math.cos(curr_time)
            # random_translation = np.random.uniform(-100, 100, 3).tolist()
            circle_transition = [x, y, 0]
            random_rotation = [0., 0., 0.]
            random_position = [circle_transition + random_rotation + [1.0]]
            self.sendIntegerEvent(random_position)

    def sendIntegerEvent(self, value):
        if self.receiver:
            print(f"<Sender> Sent value: {value}")
            self.receiver.onScriptEvent({'integer_value': value})

    def cleanup(self):
        self.running = False
        if self.timer_thread:
            self.timer_thread.join(timeout=1.0)

def createScene(rootNode):
    # Scene setup
    rootNode.addObject("VisualGrid", nbSubdiv=10, size=1000)
    rootNode.gravity = [0.0, -9.81, 0.0]
    rootNode.dt = 0.01

    # Add required plugins
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Visual")
    rootNode.addObject('RequiredPlugin', name="Sofa.GL.Component.Rendering3D")
    rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")

    # Create the sphere
    sphere = rootNode.addChild("sphere")
    sphere.addObject('MechanicalObject', name="mstate", template="Rigid3",
                     translation=[0., 0., 0.], rotation=[0., 0., 0.],
                     showObjectScale=50)

    # Visualization
    sphereVisu = sphere.addChild("VisualModel")
    sphereVisu.addObject('MeshOBJLoader', name="loader", filename="mesh/ball.obj")
    sphereVisu.addObject('OglModel', name="model", src="@loader",
                         scale3d=[50]*3, color=[0., 1., 0.])
    sphereVisu.addObject('RigidMapping')

    # Add controllers
    ball_movement_controller = rootNode.addObject(BallMovementController(mechanical_object=sphere.getObject("mstate")))
    
    # Add sender with receiver controller
    rootNode.addObject(IntegerGeneratorController(
        name="MyGenerator",
        receiverController=ball_movement_controller
    ))

    return rootNode

def main():
    import SofaRuntime
    import Sofa.Gui

    root = Sofa.Core.Node("root")
    createScene(root)
    
    Sofa.Simulation.initRoot(root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root)
    Sofa.Gui.GUIManager.SetDimension(1080, 1080)
    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()

if __name__ == '__main__':
    main()