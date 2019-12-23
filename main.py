import pytesseract
import cv2
import numpy as np
from pytesseract import Output
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from pdf2image import convert_from_path
import os
import argparse
import tempfile
import string
import pprint
import io
import base64
import html2text
import json
import sqlite3

def createPage(Q):
    page = """<html>
      <head>
      <meta name="viewport" content="initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
      </head>
      <body>
        {content}
        <script>
          var width = window.innerWidth|| document.documentElement.clientWidth|| document.body.clientWidth;
          var height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
          console.log(width, height);
          //document.body.style.backgroundSize = width + "px " + height + "px;"
          for(x of document.getElementsByTagName("span")){
            x.style.fontSize = x.style.fontSize.slice(0, -1)*width*H/W/100 + "px";
            x.style.left = x.style.left.slice(0, -1)*width/100 + "px";
            x.style.top = x.style.top.slice(0, -1)*width*H/W/100 + "px";
          }
        </script>
        <style>
          span{
            position: absolute;
            display: block;
            color: rgba(0, 0, 0, 0);
          }
          body{
            background-image: url("{background}");
            background-size: cover;
            background-repeat: no-repeat;
          }
        </style>
      </body>
    </html>
    """
    nQ = {"choices":[]}
    W, H = Q["W"], Q["H"]
    content = ""
    pcontent = f"<script>W={W};H={H};</script>"
    for t in Q["question"]:
        pcontent += f'<span style="left:{100*t[1]/W}%;top:{100*t[2]/H}%;font-size:{100*t[4]/H}%;">{t[0]}</span>\n'
    nQ["question"] = html2text.html2text(pcontent).strip()
    content += pcontent
    content += "<br/>"
    content += "<br/>"
    for c in Q["choices"]:
        pcontent = ""
        for t in c:
            pcontent += f'<span style="left:{100*t[1]/W}%;top:{100*t[2]/H}%;font-size:{100*t[4]/H}%;">{t[0]}</span>\n'
        nQ["choices"].append(html2text.html2text(pcontent).strip())
        content += pcontent + "<br/>"
    page = page.replace("{content}", content)
    page = page.replace("{background}", Q["background"])
    return (page, nQ)

def splitChoices(question, n, con):
    H, W = question.shape[:2]
    d = pytesseract.image_to_data(question, output_type=Output.DICT, lang="spa")
    n_boxes = len(d['level'])
    total_marg0 = [0, 0]
    marg0 = []
    marg_er = 1000
    for i in tqdm(range(n_boxes)):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        if d["text"][i].startswith("(") and d["text"][i].endswith(")"):
            if "a" <= d["text"][i][1:-1] <= "z":
                total_marg0 = [total_marg0[j]+[x, w+x][j] for j in range(2)]
                marg0.append([x, w+x, y, y+h, d["text"][i]])
    if len(marg0) == 0:
        return False
    avg_marg0 = [x/len(marg0) for x in total_marg0]
    total_marg1 = [0, 0]
    marg1 = []
    for m in marg0:
        e = sum([(avg_marg0[x]-m[x])**2 for x in range(2)])
        if e < marg_er:
            total_marg1 = [total_marg1[j]+m[j] for j in range(2)]
            marg1.append(m)
            #print(m, e)
    print("Question Margins", n, marg1, marg0, "\n")
    avg_marg1 = [x/len(marg1) for x in total_marg1]
    numX1 = int(avg_marg1[0])-10
    numX2 = int(avg_marg1[1])+10
    choices = question[0:H, numX1:numX2].copy()
    a, b, _ = choices.shape
    for y in range(a):
        for x in range(b):
            if (choices[y][x] != np.array([255, 255, 255])).all():
                inBox = False
                inKnownBox = False
                for i in range(n_boxes):
                    (bX, bY, bW, bH) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    if not (bX == 0 and bY == 0):
                        if (bX - 25 <= x + numX1 <= bX + bW + 25 and bY - 25 <= y <= bY + bH + 25):
                            inBox = True
                            if d["text"][i].startswith("(") and d["text"][i].endswith(")"):
                                inKnownBox = True
                if (not inBox) or (inKnownBox):
                    choices[y] = np.array([0, 0, 0])
                    break
                else:
                    choices[y] = np.array([255, 255, 255])

    lT = 30
    pC = 0
    pS = 0
    cuts = []
    for y in range(a):
        c = (choices[y][0] == np.array([255, 255, 255])).all()
        if c != pC:
            if c == 0:
                cuts.append(y)
            pS = y
            pC = c
    for y in range(a-1, -1, -1):
        if not (question[y] == np.array([255, 255, 255])).all():
            cuts.append(y)
            break

    Q = {"choices":[], "question":[]}
    with io.BytesIO() as output:
        Image.fromarray(question).save(output, format="PNG")
        contents = output.getvalue()
        Q["background"] = "data:image/png;charset=utf-8;base64, " + base64.b64encode(contents).decode()
    Q["W"] = W
    Q["H"] = H
    d = pytesseract.image_to_data(question[0:cuts[0], 0:W], output_type=Output.DICT, lang="spa")
    n_boxes = len(d['level'])
    for j in range(n_boxes):
        (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
        if int(d["conf"][j]) > 50:
            if len(d["text"][j].strip()) > 0:
                Q["question"].append([d["text"][j], x, y, w, h])


    for i in range(len(cuts)-1):
        text = []
        y1, y2 = cuts[i], cuts[i+1]
        d = pytesseract.image_to_data(question[y1:y2, int(avg_marg1[1])-10:W], output_type=Output.DICT, lang="spa")
        n_boxes = len(d['level'])
        for j in range(n_boxes):
            (x, y, w, h) = (d['left'][j], d['top'][j], d['width'][j], d['height'][j])
            if int(d["conf"][j]) > 50:
                if len(d["text"][j].strip()) > 0:
                    text.append([d["text"][j], x+int(avg_marg1[1])-10, y+y1, w, h])
        Q["choices"].append(text)
    Image.fromarray(question).save(f"{n}.pdf", "PDF" ,resolution=100.0, save_all=True)
    page, Q = createPage(Q)
    with open(f"{n}.json", "w") as f:
        json.dump(Q, f)
    sql = ''' INSERT INTO questions(question, a, b, c, d)
          VALUES(?,?,?,?,?) '''

    Q["choices"] += [" "] * (4 - len(Q["choices"]))
    con.cursor().execute(sql, (Q["question"], Q["choices"][0], Q["choices"][1], Q["choices"][2], Q["choices"][3]))
    con.commit()
    return True

def splitQuestions(page, num, con):
    path = os.path.join(tempfile.gettempdir(), f"page_{page}.png")
    r_img = cv2.imread(path)
    img = cv2.imread(path)
    H, W = img.shape[:2]

    d = pytesseract.image_to_data(img, output_type=Output.DICT, lang="spa")
    n_boxes = len(d['level'])
    total_marg0 = [0, 0]
    marg0 = []
    marg_er = 1000
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for i in tqdm(range(n_boxes)):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #cv2.putText(img, d["text"][i], (x, y), font, 1, (0, 0, 0), 2)
        """
        if int(d["conf"][i]) > 80:
            if len(d["text"][i].strip()) > 0:
                fontsize = 1
                font = ImageFont.truetype("arial", fontsize)
                while 0 < draw.textsize(d["text"][i], font=font)[0] < w:
                    fontsize += 1
                    font = ImageFont.truetype("arial", fontsize)
                    #print(fontsize, draw.textsize(d["text"][i], font=font))
                size = draw.textsize(d["text"][i], font=font)
                offset = font.getoffset(d["text"][i])
                draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255), outline=(0, 255, 0))
                draw.rectangle((x, y, x + size[0] + offset[0], y + size[1] + offset[1]), outline='black')
                draw.text((x, y), d["text"][i], font = font, fill=(0, 0, 0))
        """
        print(d["text"][i], end=" ")
        if d["text"][i].endswith("."):
            if d["text"][i][:-1].isdigit():
                total_marg0 = [total_marg0[j]+[x, w+x][j] for j in range(2)]
                marg0.append([x, w+x, y, y+h, d["text"][i]])
    #cv2.imwrite(f"{page}w.png", np.array(img_pil))
    avg_marg0 = [x/len(marg0) for x in total_marg0]
    print()
    print(marg0)
    print()
    print(total_marg0)
    print(avg_marg0)
    total_marg1 = [0, 0]
    marg1 = []
    for m in marg0:
        e = sum([(avg_marg0[x]-m[x])**2 for x in range(2)])
        if e < marg_er:
            total_marg1 = [total_marg1[j]+m[j] for j in range(2)]
            marg1.append(m)
            print(m, e)
    avg_marg1 = [x/len(marg1) for x in total_marg1]
    print(total_marg1)
    print(avg_marg1)
    numX1 = int(avg_marg1[0])-10
    numX2 = int(avg_marg1[1])+10
    cv2.rectangle(img, (numX1, 0), (numX2, H), (120, 0, 255), 3)
    nums = r_img[0:H, numX1:numX2].copy()
    #cv2.imshow('img', img)
    #cv2.waitKey(0)
    #cv2.imwrite(f"{page}p.png", img)
    #cv2.imwrite(f"{page}n.png", nums)
    print(d.keys())


    #print(d["text"])
    a, b, _ = nums.shape
    for y in range(a):
        for x in range(b):
            if (nums[y][x] != np.array([255, 255, 255])).all():
                nums[y] = np.array([0, 0, 0])
                break
        else:
            nums[y] = np.array([255, 255, 255])

    lT = 300
    pC = 0
    pS = 0
    cuts = []
    for y in range(a):
        c = (nums[y][0] == np.array([255, 255, 255])).all()
        if c != pC:
            if c == 0:
                cuts.append(y)
            pS = y
            pC = c
    for y in range(a-1, -1, -1):
        if not (r_img[y] == np.array([255, 255, 255])).all():
            cuts.append(y)
            break
    false_trues = 0
    for i in range(len(cuts)-1):
        y1, y2 = cuts[i], cuts[i+1]
        if not splitChoices(r_img[y1:y2, int(avg_marg1[0])-10:W], num+1+i-false_trues, con):
            false_trues += 1
    print(cuts)
    print(marg1)
    #cv2.imwrite(f"{page}r.png", nums)
    return len(cuts) - 1 - false_trues

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    pdf_file = args.file
    num = 0
    pages = convert_from_path(pdf_file, 500)
    open(f"{pdf_file}.db", "w").close()
    con = sqlite3.connect(f"{pdf_file}.db")

    create_table = """ CREATE TABLE IF NOT EXISTS questions (
                            id integer PRIMARY KEY,
                            question text NOT NULL,
                            a text,
                            b text,
                            c text,
                            d text
                        );
                   """

    con.cursor().execute(create_table)
    for i, page in enumerate(pages):
        page.save(os.path.join(tempfile.gettempdir(), f"page_{i+1}.png"))
        num += splitQuestions(i+1, num, con)
