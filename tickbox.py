# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:06:33 2020

@author: smanacht
"""
import tkinter as *


def open_file():
    """
    This method handles reading in Endless Sky mission files.
    It creates a mission object for each mission it finds,
    and then calls the parser to parse the data
    """
    #TODO: Add handling for mission preamble(license text)

    logging.debug("Selecting mission file...")
    f = filedialog.askopenfile()
    if f is None:  # askopenfile() returns `None` if dialog closed with "cancel".
        return
    logging.debug("Opening file: %s" % f.name)

    with open(f.name) as infile:
        mission_lines = infile.readlines()
    infile.close()

    config.mission_file_items.empty()
    parser = MissionFileParser(mission_lines)
    parser.run()

    config.active_item = config.mission_file_items.items_list[0]
    config.gui.update_option_pane()
#end open_file 