''' 
@author - Tim Schellin
2/20/2020
'''

import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import urllib
import time
from PIL import Image


IMAGES_PATH = ''
WEBCAM_IP = ''
CAMERA_DEVICE_ID = 2
MAX_DISTANCE = 0.6


def main():
	global IMAGES_PATH, WEBCAM_IP 
	IMAGES_PATH, WEBCAM_IP = load_paths()
	known_faces = load_identities()
	run_face_recognition(known_faces)


def run_face_recognition(known_faces):
	known_face_encodings = list(known_faces.values())
	known_face_names = list(known_faces.keys())

	stream = urllib.urlopen(WEBCAM_IP)
	bytes = ''
	while True:
		bytes += stream.read(1024)
		a = bytes.find(b'\xff\xd8')
		b = bytes.find(b'\xff\xd9')

		if a != -1 and b != -1:
			jpg = bytes[a:b+2]
			bytes = bytes[b+2:]
			frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
			face_locations, face_encodings = get_embed_from_img(frame, convert_to_rgb=True)

			# Loop through each face in this frame of video and see if there's a match
			for location, face_encoding in zip(face_locations, face_encodings):
				distances = face_recognition.face_distance(known_face_encodings, face_encoding)
				if np.any(distances <= MAX_DISTANCE):
					best_match_idx = np.argmin(distances)
					name = known_face_names[best_match_idx]
				else:
					name = None
				# put recognition info on the image
				draw_box_on_face(frame, location, name)
			cv2.imshow('Video', frame) # Display the resulting image
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	cv2.destroyAllWindows()


def draw_box_on_face(frame, location, name=None):
    top, right, bottom, left = location
    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)


def get_embed_from_img(image, convert_to_rgb=False):
	if convert_to_rgb:
		image = image[:, :, ::-1]
		face_locations = face_recognition.face_locations(image)
		face_encodings = face_recognition.face_encodings(image, face_locations)
		return face_locations, face_encodings


def load_identities():
	known_faces = {}
	for filename in glob.glob(os.path.join(IMAGES_PATH, '*.jpg')):
		image_rgb = face_recognition.load_image_file(filename)
		identity = os.path.splitext(os.path.basename(filename))[0]
		locations, encodings = get_embed_from_img(image_rgb, convert_to_rgb=True)
		try:
			known_faces[identity] = encodings[0]
		except:
			pass
	return known_faces


def save_to_folder(name, frame):
	img = Image.fromarray(frame)
	filename = '{0}/{1}/{1}{2}.jpg'.format(IMAGES_PATH, name, time.time())
	img.save(filename)

def load_paths():
	lines = [line.rstrip('\n') for line in open('project_paths.txt').readlines()]
	return lines[0], lines[1]

if __name__ == "__main__":
	main()
