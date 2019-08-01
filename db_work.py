import psycopg2
import sys
from PIL import Image, ImageDraw
from io import BytesIO
import io
from datetime import datetime, date, time
from cv2 import imwrite, imread
import binascii
import  PIL


def add_BD(time_b, time_e, data, value_id):
    conec_BD = psycopg2.connect(host="127.0.0.1", database="postgres", port="5432", user="postgres", password="12345")
    # imwrite('tst_to_db.png', data)
    # a = imread('tst_to_db.png')
    # with open('12390.png', "wb") as output_file:
    #     output_file.write(data)

    im = PIL.Image.fromarray(data)
    im.save("your_file.png")

    ig_open = open("your_file.png", "rb")
    img_read = ig_open.read()

    #img_read = ig_open.read()
    #binary = psycopg2.Binary(a)

    cursor_BD = conec_BD.cursor()
    cursor_BD.execute('''
        CREATE TABLE if not exists StC
        (id serial PRIMARY KEY NOT NULL,
        Time_begin timestamp NOT NULL,
        Time_end timestamp NOT NULL,
        Data bytea NOT NULL,
        Value_id text NOT NULL);''')

    cursor_BD.execute('''INSERT INTO StC (Time_begin, Time_end, Data, Value_id) VALUES (%s, %s, %s, %s);''',
                      (time_b, time_e, psycopg2.Binary(img_read), value_id))
    conec_BD.commit()
    cursor_BD.close()


def dwd_data(utc_dt_b, utc_dt_e, type):
    conec_BD = psycopg2.connect(host="127.0.0.1", database="postgres", port="5432", user="postgres", password="12345")

    if True:
        cursor = conec_BD.cursor()
        sl_v = ''
        sl_io = {
            1: 'Input',
            2: 'Output'
        }
        if (type == 'Any'):
            print('successfully')
            # d_begin = str(input())
            # d_end = str(input())
            date_begin = datetime.strftime(utc_dt_b, '%Y-%m-%d %H:%M:%S')
            date_end = datetime.strftime(utc_dt_e, '%Y-%m-%d %H:%M:%S')
            cursor.execute(
                '''SELECT Time_begin, Time_end, Data FROM StC WHERE Time_begin BETWEEN %s AND %s; ''',
                (date_begin, date_end))
        else :
            print('successfully')
            # d_begin = str(input())
            # d_end = str(input())
            date_begin = datetime.strftime(utc_dt_b, '%Y-%m-%d %H:%M:%S')
            date_end = datetime.strftime(utc_dt_e, '%Y-%m-%d %H:%M:%S')
            cursor.execute(
                '''SELECT Time_begin, Time_end, Data FROM StC WHERE Value_id = %s AND Time_begin BETWEEN %s AND %s; ''',
                (type.__str__(), date_begin, date_end))
            # results = cursor.fetchone()

    else:
        print("Error")

    count = 0
    for row in cursor:
        count+=1
        S1 = str(row[0]).replace(':', '.')
        file_name = S1 + '.png'
        with open('frames\\'+file_name, "wb") as output_file:
            output_file.write(row[2])
    cursor.close()
    conec_BD.close()

    Text_val = 'Промежуток времени от : '+ date_begin + ' до ' + date_end +'\n' + ' Тип: '+ type.__str__() +'\n'
    Text_val+= 'Строчек: '+ count.__str__() +'\n'
    Text_val+= 'Путь: "\TensorFlowProject3\frames"' +'\n'
    return  Text_val
