import numpy as np

class BDTracker(object):
    count = 0
    def __init__(self, img_id, img_file, bbox, score) :
        self.obj_id = BDTracker.count
        BDTracker.count += 1
        self.history_objservation = [bbox] # for bbox[cx, cy, w, h]
        self.velocity = None
        self.history_id = [img_id] # for img_id
        self.history_path = [img_file] # for img_file
        self.history_score = [score]
        self.status = 'in'

    def update (self, img_id, img_file, bbox, score) :
        self.history_objservation.append(bbox) # for bbox
        self.history_id.append(img_id)# for img_id
        self.history_path.append(img_file) # for img_file
        self.history_score.append(score)
        if len(self.history_objservation) > 1 :
            step = self.history_id[-1] - self.history_id[-2]
            start_x, start_y = self.history_objservation[-2][:2]
            end_x, end_y = self.history_objservation[-1][:2]
            velocity = [(end_x - start_x)/step, (end_y - start_y)/step]

            if np.linalg.norm(velocity) >= 2  : # 2 #####<--the minimum value for movig objects
                self.velocity = velocity
            else :
                self.velocity = None

        if not self.velocity == None :
            assumed_center = self.forecast(img_id+1)
            if (assumed_center[0][0] - 128)**2 + (assumed_center[0][1] - 128)**2 > 128**2 :
                self.status = 'out'

    def forecast (self, img_id) :
        if not self.velocity == None :
            step = img_id - self.history_id[-1]
            origin_x, origin_y = self.history_objservation[-1][:2]
            vel_x, vel_y = self.velocity
            next_x = origin_x + vel_x * step
            next_y = origin_y + vel_y * step
            return np.array([[next_x, next_y]])
        else :
            return np.array([[-1, -1]])

