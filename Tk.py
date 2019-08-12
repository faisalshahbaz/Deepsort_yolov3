#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tkinter as tk
ON = 1
OFF = 0


class Gui():
    def __init__(self):

        self.ifspeed = OFF
        self.iffall = OFF
        self.ifregion = OFF
        self.ifline = OFF
        self.ifsingle_cross = OFF  # 默认为双向穿越
        self.ifreverse = OFF
        self.ifsave = OFF
        self.pathToLoad = "/home/tom/桌面/行人检测算法/people/003.avi"
        self.pathToSave = "./alarm_frame/"
        self.speedMax = 10
        self.speedMin = 2

    def gui(self):
        # 作用： 绘制GUI界面并且获取参数

        root = tk.Tk()  # 创建窗口对象的背景色
        Bool_track = tk.IntVar()
        Bool_fall = tk.IntVar()
        Bool_region = tk.IntVar()
        Bool_line = tk.IntVar()
        Bool_single_line = tk.IntVar()
        Bool_reverse = tk.IntVar()
        Bool_save = tk.IntVar()

        S_entry = tk.StringVar()
        S_save = tk.StringVar()
        S_speed = tk.StringVar()

        C_track = tk.Checkbutton(root,
                                 text="Does it has speed limit?",
                                 variable=Bool_track,
                                 onvalue=ON,
                                 offvalue=OFF,
                                 height=10,
                                 width=30)  # if tracking?

        C_fall = tk.Checkbutton(root,
                                text="Does it judge fall?",
                                variable=Bool_fall,
                                onvalue=ON,
                                offvalue=OFF,
                                height=10,
                                width=30)  # if detect falling?
        C_region = tk.Checkbutton(root,
                                  text="Does it has warning area?",
                                  variable=Bool_region,
                                  onvalue=ON,
                                  offvalue=OFF,
                                  height=10,
                                  width=30)  # if have alert area?
        C_line = tk.Checkbutton(root,
                                text="Does it has warning line?",
                                variable=Bool_line,
                                onvalue=ON,
                                offvalue=OFF,
                                height=10,
                                width=30)  # if have alert line?
        C_single_line = tk.Checkbutton(root,
                                       text="Single cross? Double by default.",
                                       variable=Bool_single_line,
                                       onvalue=ON,
                                       offvalue=OFF,
                                       height=10,
                                       width=30)  # if single cross?
        C_reverse = tk.Checkbutton(root,
                                   text="Reverse direction?",
                                   variable=Bool_reverse,
                                   onvalue=ON,
                                   offvalue=OFF,
                                   height=10,
                                   width=30)  # if single cross?
        C_save = tk.Checkbutton(root,
                                text="Save videos?",
                                variable=Bool_save,
                                onvalue=ON,
                                offvalue=OFF,
                                height=10,
                                width=30)  # if save?

        # B_finish = tk.Button(root, text="提交", command=root.quit())
        tk.Label(root, text='File path of opening').pack()
        tk.Entry(root, textvariable=S_entry,
                 width=30).pack()  # address of file
        C_track.pack()
        tk.Label(root, text='Maximum speed').pack()
        tk.Entry(root, textvariable=S_speed, width=30).pack()  # max speed
        C_fall.pack()
        C_region.pack()
        C_line.pack()
        C_single_line.pack()
        C_reverse.pack()
        C_save.pack()
        tk.Label(root, text='File path of saving', width=30).pack()
        tk.Entry(root, textvariable=S_save,
                 width=30).pack()  # address of writing file

        tk.Button(root, text="Finish!", command=root.quit,
                  height=10).pack()  # command只写函数名！

        root.mainloop()  # 进入消息循环

        self.ifspeed = Bool_track.get()
        self.iffall = Bool_fall.get()
        self.ifregion = Bool_region.get()
        self.ifline = Bool_line.get()
        self.ifsingle_cross = Bool_single_line.get()  # OFF 为双向穿越
        self.ifreverse = Bool_reverse.get()  # 横穿的方向是否改变
        self.ifsave = Bool_save.get()

        Path_load = S_entry.get()
        if len(Path_load) != 0:
            self.pathToLoad = Path_load

        Path_save = S_save.get()
        if len(Path_save) != 0:
            self.pathToSave = Path_save

        speed = S_speed.get()
        if len(speed) != 0:
            self.speedMax = int(speed)
