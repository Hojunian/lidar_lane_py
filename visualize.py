import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import vispy
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera
from vispy.scene import SceneCanvas

import pickle

#######################################################
#Specify frame sequence root directory by --data_folder parameter (ie. --data_folder=./001)
#and the script will let you iterate over the frames in the specified sequence.

#Using keys "N" and "B" switch between frames.
#Using keys "T" and "D" switch between visualizations
#for current frame: 
#cloud with points colored by intensity <-> lane labels.

#The window is interactive: wheel: zoom in/out, left hold left mouse button and move to rotate.
########################################################

class Vis():
    def __init__(self, data_folder, show_grid=False, no_color=False):
        self.index = 0 #frame index in the sequence
        self.state = 0 #display mode, 0-cloud, 1-road_label

        self.show_grid = show_grid
        self.no_color = no_color 

        self.lidar_paths, self.lane_paths = self.read_data(data_folder)

        self.reset()
        self.visualize_cloud()

    def reset(self):
        self.canvas = SceneCanvas(keys='interactive',
                                  show=True,
                                  size=(1600, 900))
        self.canvas.events.key_press.connect(self._key_press)
        self.canvas.events.draw.connect(self._draw)

        self.grid = self.canvas.central_widget.add_grid()
        self.scan_view = vispy.scene.widgets.ViewBox(parent=self.canvas.scene,
                                                     camera=TurntableCamera(distance=80.0))

        self.grid.add_widget(self.scan_view)
        if self.show_grid:
            self.grid_l = vispy.scene.visuals.GridLines(parent=self.scan_view.scene, color=(1, 1, 1))
            self.grid_l.transform = vispy.visuals.transforms.MatrixTransform()
            self.grid_l.transform.translate(np.asarray(self.canvas.size) / 2)

        self.scan_vis = visuals.Markers()
        self.scan_view.add(self.scan_vis)
        visuals.XYZAxis(parent=self.scan_view.scene)

        self.lane_line = vispy.scene.visuals.Line(parent=self.scan_view.scene)
        self.reset_road()

    def reset_road(self):
        self.lane_line.visible = False

    def read_data(self, data_folder):
        get_full_paths = lambda path: [os.path.join(path, f) for f in sorted(os.listdir(path))]

        def get_label_paths(sub_dir):
            full_dir_path = os.path.join(data_folder, sub_dir) 
            if os.path.isdir(full_dir_path):
                return get_full_paths(full_dir_path)
            else:
                return None

        lidar_paths = get_label_paths("lidar")
        lane_paths  = get_label_paths("road_label/lane")

        return lidar_paths, lane_paths

    def point_color_by_intensity(self, points):
        scale_factor = 10
        scaled_intensity = np.clip(points[:, 3] * scale_factor, 0, 255)
        scaled_intensity = scaled_intensity.astype(np.uint8)
        cmap = plt.get_cmap("viridis")

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        color_range = color_range.reshape(256, 3).astype(np.float32) / 255.0
        colors = color_range[scaled_intensity]
        return colors

    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)
        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)
        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]
        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def visualize_cloud(self):
        if self.lidar_paths:
            print(self.lidar_paths)
            lidar_path = self.lidar_paths[self.index]
            print(self.index)
            print(lidar_path)
        else:
            print("No point cloud data")
            return

        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
        if self.no_color:
            colors=[1,1,1]
        else:
            colors = self.point_color_by_intensity(points)

        self.canvas.title = f"Frame: {self.index} / {len(self.lidar_paths)} - {self.lidar_paths[self.index]}"
        self.scan_vis.set_data(points[:, :3],
                                             face_color=colors,
                                             edge_color=colors,
                                             size=1.0)

    def visualize_road_labels(self):
        self.visualize_cloud()
        self.canvas.title = f"Frame (Lane labels): {self.index} / {len(self.lidar_paths)} - {self.lidar_paths[self.index]}"

        def get_connection(lines):
            _connect = []
            last = 0
            for l in lines:
                n = l.shape[0]
                _connect += [[last+i, last+i+1] for i in range(n - 1)]
                last += n
            return np.array(_connect)

        #draw lanes
        if self.lane_paths:
            with open(self.lane_paths[self.index], 'rb') as f:
                lanes = pickle.load(f)
            if lanes:
                self.lane_line.set_data(pos=np.concatenate(lanes, axis=0),
                                        connect=get_connection(lanes),
                                        color=np.array([1, 0, 0]))
                self.lane_line.visible = True
            else:
                print("The lanes data file is empty.")
        else:
            print("No lane data exist for the cloud.")

    def _key_press(self, event):
        if event.key == 'N':
            if self.index < len(self.lidar_paths) - 1:
                self.index += 1
        if event.key == 'B':
            if self.index > 0:
                self.index -= 1
        if event.key == 'T':
            self.state = min(self.state+1, 1) 
        if event.key == 'D':
            self.state = max(self.state-1, 0)
        if event.key == 'Q' or event.key == 'Escape':
            self.destroy()

        self.reset_road()
        if self.state == 0:
            self.visualize_cloud()
        elif self.state == 1:
            self.visualize_road_labels()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        vispy.app.quit()

    def _draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()

    def run(self):
        self.canvas.app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./017', help='Path to the sequence root directory')
    parser.add_argument('--show_grid', action='store_true') 
    parser.add_argument('--no_color', action='store_true', help='Display frame points in white color. If not checked, points are colored by intensity') 
    opts = parser.parse_args()

    if not os.path.isdir(opts.data_folder):
        print("Couldn't open the cloud sequence root folder: " + opts.data_folder)
        exit()

    vis = Vis(opts.data_folder, opts.show_grid, opts.no_color)
    vis.run()
