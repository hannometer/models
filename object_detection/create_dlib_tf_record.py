# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert DLIB imglab dataset to TFRecord for object_detection.

Example usage:
    ./create_dlib_tf_record --xml=/home/user/dlib-dataset.xml \
        --output_path=/home/user/dlib.record

It will also create a dlib_label_map.pbtxt file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import xml.etree.ElementTree

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('xml', '', 'imglab XML file with object annotations.')
flags.DEFINE_string('set', 'train', 'Convert training set, validation set or merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations', '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
FLAGS = flags.FLAGS

def find_all_labels_recursive( xml ):
    labels = set()
    for child in xml:
        child_result = find_all_labels_recursive(child)
        if child.tag == 'label':
          labels.add(child.text)
        labels.update(child_result)
    return labels

def main(_):
    xml_file = FLAGS.xml

    logging.info('Reading from DLIB %s dataset.', xml_file)

    # find all labels in xml
    xml_data = xml.etree.ElementTree.parse( xml_file ).getroot()
    labels = find_all_labels_recursive(xml_data)
    labels = sorted(labels)
    print(labels)

    # create pbtxt
    label_map_filename = os.path.splitext(FLAGS.output_path)[0] + '.pbtxt'
    with open(label_map_filename,'w') as f:
        for idx,lbl in enumerate(labels):
            f.write("item {{\n  id: {}\n  name: '{}'\n}}\n\n".format(idx+1,lbl))

    # create TF record file
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    xml_file_dir = os.path.dirname(xml_file)
    print('XML file directory is {}.'.format(xml_file_dir))
    for img_node in xml_data.find('images'):

        # get image path
        img_filename = img_node.attrib['file']
        if os.path.isabs( img_filename ):
            img_path = img_filename
        else:
            img_path = os.path.join(xml_file_dir,img_filename)

        # read image
        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()
        width,height = image.size

        print("Image {}: {}x{}".format(os.path.basename(img_filename),width,height))

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        difficult_obj = []

        # read annotations
        for annot_node in img_node:
            label_node = annot_node.find('label')
            if label_node is not None:
                label = label_node.text
            else:
                print("Skipping annotation because label is empty")

            bbox = dict(ymin=int(annot_node.attrib['top']),
                        xmin=int(annot_node.attrib['left']),
                        xmax=int(annot_node.attrib['left'])+int(annot_node.attrib['width']),
                        ymax=int(annot_node.attrib['top'])+int(annot_node.attrib['height']))

            is_truncated = bbox['xmin'] < 0 or bbox['xmax'] >= width or bbox['ymin'] < 0 or bbox['ymax'] >= height
            is_difficult = False

            xmin.append(float(bbox['xmin']) / width)
            ymin.append(float(bbox['ymin']) / height)
            xmax.append(float(bbox['xmax']) / width)
            ymax.append(float(bbox['ymax']) / height)
            classes_text.append(label.encode('utf8'))
            classes.append(labels.index(label)+1)
            truncated.append(int(is_truncated))
            difficult_obj.append(int(is_difficult))

        if len(classes) > 0:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/filename': dataset_util.bytes_feature(img_filename.encode('utf8')),
                'image/source_id': dataset_util.bytes_feature(img_filename.encode('utf8')),
                'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
                'image/encoded': dataset_util.bytes_feature(encoded_jpg),
                'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
                'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
                'image/object/class/label': dataset_util.int64_list_feature(classes),
                'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
                'image/object/truncated': dataset_util.int64_list_feature(truncated),
                #'image/object/view': dataset_util.bytes_list_feature(poses),
            }))
            writer.write(example.SerializeToString())

    # print(data)
    #examples_list = dataset_util.read_examples_list(examples_path)
    # for idx, example in enumerate(examples_list):
    #   if idx % 100 == 0:
    #     logging.info('On image %d of %d', idx, len(examples_list))
    #   path = os.path.join(annotations_dir, example + '.xml')
    #   with tf.gfile.GFile(path, 'r') as fid:
    #     xml_str = fid.read()
    #   xml = etree.fromstring(xml_str)
    #   data = dataset_util.recursive_parse_xml_to_dict(xml)
    #   tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
    #                                   FLAGS.ignore_difficult_instances)
    #   writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
  tf.app.run()
