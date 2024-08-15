# The script converts raw measurement data on the lens and the reference plane (here referred as ring) that is in the form of cylindrical coordinates i.e. angle, height and radius.
#It coverts that coordinate data into cartesian coordinates. Then planes are fit using least square method into each lens and ring data.
#And then angle between these planes is measure by measuring angle between normal vectors of these planes.
import numpy as np # Importing the NumPy library for numerical operations
import csv # Importing the csv module to handle CSV file operations
from scipy.optimize import least_squares # Importing the least_squares function from scipy for optimization
from math import cos, sin, radians, acos # Importing trigonometric functions from the math module
import plotly.graph_objects as go # Importing Plotly for 3D visualizations

def read_combined_coordinates(filename):
    with open(filename,'r') as file: # Opening the CSV file in read mode
        reader = csv.reader(file) # Creating a CSV reader object
    next(reader) # Skipping the header row
    return [(float(angle), float(ring_height), float(lens_height)) for angle, ring_height, lens_height in reader] # Returning a list of tuples containing angles and heights as floats

def cylindrical_to_cartesian(r, theta, z):
    x = r * cos(theta) # Calculating the x-coordinate
    y = r * sin(theta) # Calculating the y-coordinate
    return [x, y, z] # Returning the Cartesian coordinates as a list

def fit_plane(points):
    """Fit a plane to a set of points using least squares."""
    def plane_equation(params, x, y):
        a, b, c = params # Unpacking the plane parameters
        return a * x + b * y + c # Defining the plane equation

    def residuals(params, coordinates, z):
        return plane_equation(params, coordinates[:,0], coordinates[:,1]) - z # Calculating residuals for least squares

    xy = np.array(points)[:, :2] # Extracting x and y coordinates from the points
    z = np.array(points)[:, 2] # Extracting z coordinates
    initial_guess = [1, 1, 1] # Setting an initial guess for the plane parameters
    res = least_squares(residuals, initial_guess, args=(xy, z)) # Performing the least squares fitting
    return res.x # Returning the plane parameters a, b, c

def angle_between_planes(plane1, plane2):
    """Calculate the angle between two planes given their coefficients."""
    normal1 = np.array([-plane1[0], -plane1[1], 1]) # Calculating the normal vector for the first plane
    normal2 = np.array([-plane2[0], -plane2[1], 1]) # Calculating the normal vector for the second plane
    cos_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)) # Calculating the cosine of the angle between the normals
    return np.degrees(acos(np.clip(cos_angle, -1.0, 1.0))) # Returning the angle in degrees, using acos and clipping for numerical stability

def generate_plane_points(plane_coeff, x_range, y_range):
    """Generate points on a plane for a given x and y range."""
    a, b, c = plane_coeff # Unpacking the plane coefficients
    x, y = np.meshgrid(x_range, y_range) # Creating a mesh grid of x and y values
    z = a * x + b * y + c # Calculating the z values for the plane
    return x, y, z # Returning the x, y, and z arrays

def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3) # Normalizing and reshaping the vectors
    v = np.cross(a, b) # Calculating the cross product of a and b
    c = np.dot(a, b) # Calculating the dot product of a and b
    s = np.linalg.norm(v) # Calculating the norm of the cross product
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) # Creating the skew-symmetric cross-product matrix
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2)) # Returning the rotation matrix

def transform_points(points, transformation_matrix):
    """Apply a transformation matrix to a set of points."""
    return np.dot(points, transformation_matrix.T) # Transforming the points using the matrix and returning the result

def main():
    # Specifying the file paths for input and output CSV files
    combined_file = r'input-file-path' #csv file with columns: angle, righ height, lens height
    output_file = r'output-file-path' # csv file that will contain columns: a, b and c coefficients that define a plane of ring and lens and the angle between the planes
    ring_radius = 36 # Specifying the ring radius in mm
    lens_radius = 25 # Specifying the lens radius in mm

    combined_data = read_combined_coordinates(combined_file) # Reading the combined coordinate data

    # Converting cylindrical coordinates to Cartesian coordinates for the ring and lens
    ring_points = [cylindrical_to_cartesian(ring_radius, radians(angle), height) for angle, height, _ in combined_data]
    lens_points = [cylindrical_to_cartesian(lens_radius, radians(angle), height) for angle, _, height in combined_data]

    # Fitting planes to the ring and lens points
    plane_ring = fit_plane(ring_points)
    plane_lens = fit_plane(lens_points)

    # Calculating the angle between the two fitted planes
    angle = angle_between_planes(plane_ring, plane_lens)

    with open(output_file, 'w', newline='') as file: # Opening the output CSV file in write mode
        writer = csv.writer(file) # Creating a CSV writer
        writer.writerow(['Plane Ring', 'Plane Lens', 'Angle Between Planes']) # Writing the header row
        writer.writerow([plane_ring, plane_lens, angle]) # Writing the plane coefficients and the angle

    # Calculating the normal vector of the ring plane and normalizing it
    normal_ring = np.array([-plane_ring[0], -plane_ring[1], 1])
    normal_ring /= np.linalg.norm(normal_ring)

    # Calculating the rotation matrix to align the ring plane with the horizontal
    rot_matrix = rotation_matrix_from_vectors(normal_ring, np.array([0, 0, 1]))

    # Applying the rotation matrix to both ring and lens points
    transformed_ring_points = transform_points(np.array(ring_points), rot_matrix)
    transformed_lens_points = transform_points(np.array(lens_points), rot_matrix)

    # Defining common ranges for X and Y axes for visualization
    x_range = np.linspace(-50, 50, 20)
    y_range = np.linspace(-50, 50, 20)

    # Generating points on the transformed ring and lens planes for visualization
    x_ring, y_ring, z_ring = generate_plane_points(fit_plane(transformed_ring_points), x_range, y_range)
    x_lens, y_lens, z_lens = generate_plane_points(fit_plane(transformed_lens_points), x_range, y_range)

    fig = go.Figure() # Creating a figure object

    # Adding the transformed ring plane to the visualization
    fig.add_trace(go.Surface(x=x_ring, y=y_ring, z=z_ring, name='Ring Plane (Transformed)', opacity=0.5))

    # Adding the transformed lens plane to the visualization
    fig.add_trace(go.Surface(x=x_lens, y=y_lens, z=z_lens, name='Lens Plane (Transformed)', opacity=0.5))

    # Updating the layout of the figure with titles and axis labels
    fig.update_layout(title='Transformed Ring and Lens Planes', autosize=True,
                      scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))

    fig.show() # Displaying the figure

if __name__ == "__main__":
    main() # Running the main function if this script is the main program

print("done!") # Printing "done!" after the main function execution
