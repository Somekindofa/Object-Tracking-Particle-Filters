#%%
from distutils.log import Log
import numpy as np
import cv2
import matplotlib.pyplot as plt

path_to_video = "C:\\Users\\The Beast\\OneDrive\\ENSEA\\TVI\\TP\\TP3-4\\video sequences\\synthetic\\escrime-4-3.avi"
nb_bins = 8
N = 35 # Number of Particle Points
lambd = 0.9

def get_frame(video, frame_number): # return the frame at the given frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, frame = video.read() # frame is a 3D array of size (height, width, 3)
    return frame

def get_frame_nbr(video):
    video = cv2.VideoCapture(path_to_video)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames

def get_rect(frame): # return the rectangle selected by the user
    rect = cv2.selectROI(frame)
    return rect

def get_rect_center(rect): # return the center of the rectangle
    x, y, w, h = rect
    return (x + int(np.floor(w/2)), y + int(np.floor(h/2)))

def calc_hist(rect):
    # Open the video
    video = cv2.VideoCapture(path_to_video)
    # Get the first frame
    frame = get_frame(video, 0)
    # Get the rectangle
    (x, y, w, h) = rect
    # Get the ROI
    roi = frame[y:y+h, x:x+w]

    # Separate the channels
    b, g, r = cv2.split(roi)

    # Calculate the histogram for each channel
    hist_b = cv2.calcHist([b], [0], None, [nb_bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [nb_bins], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [nb_bins], [0, 256])

    # Normalize the histograms
    hist_b = hist_b.flatten() / np.sum(hist_b)
    hist_g = hist_g.flatten() / np.sum(hist_g)
    hist_r = hist_r.flatten() / np.sum(hist_r)

    # Stack the histograms for each channel
    hist = np.vstack((hist_b, hist_g, hist_r))

    # Average the channels of the histogram
    hist = np.mean(hist, axis=0).reshape(1, nb_bins)

    # Close the video
    video.release()
    # Close all the windows
    cv2.destroyAllWindows()

    return hist


def calc_hist_bbox(bbox, frame_nbr = 0):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    # Open the video
    video = cv2.VideoCapture(path_to_video)
    # Get the frame
    frame = get_frame(video, frame_nbr)
    # Get the ROI
    roi = frame[y:y+h, x:x+w]
    b, g, r = cv2.split(roi)
    # Calculate the histogram for each channel
    hist_b = cv2.calcHist([b], [0], None, [nb_bins], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [nb_bins], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [nb_bins], [0, 256])

    # Normalize the histograms
    hist_b = hist_b.flatten() / np.sum(hist_b)
    hist_g = hist_g.flatten() / np.sum(hist_g)
    hist_r = hist_r.flatten() / np.sum(hist_r)
    # Stack the histograms for each channel
    hist = np.vstack((hist_b, hist_g, hist_r))
    # Average the channels of the histogram
    hist = np.mean(hist, axis=0).reshape(1, nb_bins)
    # Close the video
    video.release()
    # Close all the windows
    cv2.destroyAllWindows()

    return hist # shape : (1, nb_bins)


dist_hist = np.zeros(N)

def distance_arr(hist_r, hist_list):
    # Compute the distance for each channel
    dist_channel = np.sqrt(1 - np.sum(np.sqrt(hist_r * hist_list), axis=1))
    # Compute the average distance across all channels
    dist_hist = np.mean(dist_channel)
    return dist_channel


# this function returns the resampled particles and the number of particles i at frame k
def systematic_resampling(weights, particles):
    # Compute the cumulative sum of the weights
    cum_sum_weights = np.cumsum(weights)
    # Generate N equally spaced numbers between 0 and 1
    u = np.linspace(0, 1, N, endpoint=False)
    # Initialize the index of the resampled particles
    N_i = np.zeros(N, dtype=int)
    # Initialize the index of the particles
    i = 0
    # Initialize the index of the cum_sum_weights
    j = 0
    # Resample the particles
    while i < N:
        if u[i] < cum_sum_weights[j]:
            N_i[i] = j
            i += 1
        else:
            j += 1
    # Resample the particles
    resampled_particles = particles[N_i]
    return resampled_particles, N_i

def survival_of_the_fittest(weights, particles):
    # Sort the particles by their weights
    sorted_particles = particles[np.argsort(weights)]
    # Get the N/2 best particles
    resampled_particles = sorted_particles[int(N/2):]
    # Generate N/2 new particles with mutations
    for i in range(int(N/2)):
        resampled_particles = np.vstack((resampled_particles, np.random.randint(0, 256, (1, 2))))
    return resampled_particles

def proportional_resampling(weights):
    N_i = np.random.choice(N, N, p=weights)
    resampled_particles = particles[N_i]
    return resampled_particles, N_i


def g(distance, lamb = 0.2): # Likelihood function. 
    print("distance:", distance)
    ret = np.exp(-lamb*distance**2)
    print("ret:", ret)
    return ret

def particle_filter_update(resampled_particles, hist_ref, hist_list, weights):
    # Compute the distance for each channel
    dist_channel = np.sqrt(1 - np.sum(np.sqrt(hist_ref * hist_list), axis=1))
    # Compute the average distance across all channels
    dist_hist = np.mean(dist_channel)
    # Update the weights
    for i in range(N):
        weights[i] = weights[i] * g(dist_hist, 0.5)
    # Normalize the weights
    weights = weights / np.sum(weights)
    return weights


# during the prediction step, the particles bounded boxes are averaged. 
def prediction_step(particles, bbox, weights):
    # Average the particles bounded boxes along the x and y axis
    weighted_avg_bbox = np.zeros(4)
    for i in range(N):
        weighted_avg_bbox[0] += bbox[i, 0] * weights[i] # x
        weighted_avg_bbox[1] += bbox[i, 1] * weights[i] # y
        weighted_avg_bbox[2] += bbox[i, 2] * weights[i] # w
        weighted_avg_bbox[3] += bbox[i, 3] * weights[i] # h
    weighted_avg_bbox = weighted_avg_bbox.astype(int)
    
    # Generate new particles around the center of the weighted average bounding box.
    for i in range(N):
        particles[i, 0] = np.random.randint(weighted_avg_bbox[0], weighted_avg_bbox[2])
        particles[i, 1] = np.random.randint(weighted_avg_bbox[1], weighted_avg_bbox[3])
        
    return particles










#%% Make it all work together
# Open the video
video = cv2.VideoCapture(path_to_video)
# Get the first frame
frame = get_frame(video, 0)
# Get the rectangle
rect = get_rect(frame)
# Get the center of the rectangle
rect_center = get_rect_center(rect)
# Get the size of the rectangle
rect_size = (rect[2], rect[3])
# Close the video
video.release()
# Close all the windows
cv2.destroyAllWindows()
hist_ref = calc_hist(rect) # shape (1, nb_bins)

# Open the video
video = cv2.VideoCapture(path_to_video)

# Initialize weights uniformly
weights = np.ones(N) / N
frame_number = 0
particles = np.zeros((N, 2))
for i in range(N):
    particles[i, 0] = np.random.randint(rect_center[0] - rect_size[0] / 2, rect_center[0] + rect_size[0] / 2)
    particles[i, 1] = np.random.randint(rect_center[1] - rect_size[1] / 2, rect_center[1] + rect_size[1] / 2)


while frame is not None:
    # Add a bounding box around each pixel particle.
    bbox = np.zeros((N, 4), dtype=int)
    for i in range(N):
        bbox[i, 0] = int(particles[i, 0]) - int(rect_size[0] / 2)
        bbox[i, 1] = int(particles[i, 1]) - int(rect_size[1] / 2)
        bbox[i, 2] = int(particles[i, 0]) + int(rect_size[0] / 2)
        bbox[i, 3] = int(particles[i, 1]) + int(rect_size[1] / 2)

    # get histogram for each bounded box
    hist_list = np.zeros((N, nb_bins))
    for i in range(N):
        hist_list[i] = calc_hist_bbox(bbox[i])

    # Directly perform the weight update
    for i in range(N):
        weights[i] = weights[i] * g(channel_avg_dist[i], 0.5)

    # Normalize the weights
    weights = weights / np.sum(weights)

    # Visualization (Optional)
    # Draw bounding box based on particle weights
    weighted_avg_bbox_x = int(np.average(bbox[:, 0], weights=weights))
    weighted_avg_bbox_y = int(np.average(bbox[:, 1], weights=weights))
    weighted_avg_bbox_w = int(np.average(bbox[:, 2] - bbox[:, 0], weights=weights))
    weighted_avg_bbox_h = int(np.average(bbox[:, 3] - bbox[:, 1], weights=weights))

    cv2.rectangle(frame, (weighted_avg_bbox_x, weighted_avg_bbox_y),
                (weighted_avg_bbox_x + weighted_avg_bbox_w, weighted_avg_bbox_y + weighted_avg_bbox_h), (0, 255, 0), 2)

    cv2.circle(frame, (weighted_avg_bbox_x, weighted_avg_bbox_y), 5, (0, 0, 255), -1)
    # Display the frame with the bounding box
    cv2.imshow("Particle Filter Tracking", frame)

    # Move to the next frame
    frame_number += 1
    frame = get_frame(video, frame_number)
    

    # Check for key press to exit the loop
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the video capture object and close all windows
video.release()
cv2.destroyAllWindows()

