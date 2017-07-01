# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 19:10:49 2017

@author: Jelena
"""

import time,sys
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

class MyHandler(PatternMatchingEventHandler):
    patterns = ["*.jpg", "*.png"]
    
    def process (self, event):
        """
        event.event_type 
            'modified' | 'created' | 'moved' | 'deleted'
        event.is_directory
            True | False
        event.src_path
            path/to/observed/file
        """
        name = event.src_path
        lhd, rhs = name.split("\\", 1)
        from handle import visual
        visual(self, name)
        
    # the file will be processed there
    def on_created(self, event):
        self.process(event)
      
        
if __name__ == '__main__':
    args = sys.argv[1:]
    observer = Observer()
    observer.schedule(MyHandler(), path=args[0] if args else '.')
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()