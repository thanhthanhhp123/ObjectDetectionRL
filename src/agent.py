from src.models import *
from src.tools import *

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import tqdm


class Agent(object):
    def __init__(self, classes, alpha = 0.2, nu = 3.0, threshold = 0.5, episodes = 100,
                 backbone = 'vgg16', test = False, save_path = 'models/'):
        self.n_actions = 9
        h, w = 224, 224
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        self.epsilon = 1
        self.alpha = alpha
        self.nu = nu
        self.threshold = threshold
        self.actions_history = []
        self.steps_done = 0

        self.classes = classes

        self.extractor = FeatureExtractor(network=backbone).to(self.device)
        self.extractor.eval()

        if not test:
            self.policy_net = DQN(h, w, self.n_actions, 9).to(self.device)
        else:
            self.policy_net = self.load()

        self.target_net = DQN(h, w, self.n_actions, 9).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(10000)
        self.batch_size = 32
        self.episodes = episodes
        self.target = 1
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)

        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)



    def save(self):
        torch.save(self.policy_net.state_dict(), self.save_path + 'model.pth')
    
    def load(self):
        return torch.load(self.save_path + 'model.pth')
    
    def IOU(self, box1, box2):
        try:
            box1_left, box1_right, box1_top, box1_bottom = box1
        except:
            print('Box1:', box1)
        box2_left, box2_right, box2_top, box2_bottom = box2
        
        inter_top = max(box1_top, box2_top)
        inter_left = max(box1_left, box2_left)
        inter_bottom = min(box1_bottom, box2_bottom)
        inter_right = min(box1_right, box2_right)
        inter_area = max(((inter_right - inter_left) * (inter_bottom - inter_top)), 0)
        
        box1_area = (box1_right - box1_left) * (box1_bottom - box1_top)
        box2_area = (box2_right - box2_left) * (box2_bottom - box2_top)
        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area
        return iou
    
    def compute_reward(self, actual_state, previous_state, ground_truth):
        res = self.IOU(actual_state, ground_truth) - self.IOU(previous_state, ground_truth)
        if res <= 0:
            return -1
        return 1
    
    def compute_trigger_reward(self, actual_state, ground_truth):
        if self.IOU(actual_state, ground_truth) > self.threshold:
            return self.nu
        return -1 * self.nu
    
    def calculate_box_position(self, current_coord, action):
        real_x_min, real_x_max, real_y_min, real_y_max = current_coord

        alpha_h = self.alpha * (real_y_max - real_y_min)
        alpha_w = self.alpha * (real_x_max - real_x_min)
        if action == 1: # Right
            real_x_min += alpha_w
            real_x_max += alpha_w
        if action == 2: # Left
            real_x_min -= alpha_w
            real_x_max -= alpha_w
        if action == 3: # Up 
            real_y_min -= alpha_h
            real_y_max -= alpha_h
        if action == 4: # Down
            real_y_min += alpha_h
            real_y_max += alpha_h
        if action == 5: # Bigger
            real_y_min -= alpha_h
            real_y_max += alpha_h
            real_x_min -= alpha_w
            real_x_max += alpha_w
        if action == 6: # Smaller
            real_y_min += alpha_h
            real_y_max -= alpha_h
            real_x_min += alpha_w
            real_x_max -= alpha_w
        if action == 7: # Fatter
            real_y_min += alpha_h
            real_y_max -= alpha_h
        if action == 8: # Taller
            real_x_min += alpha_w
            real_x_max -= alpha_w
                
        real_x_min = self.rewrap(real_x_min)
        real_x_max = self.rewrap(real_x_max)
        real_y_min = self.rewrap(real_y_min)
        real_y_max = self.rewrap(real_y_max)
        
        return [real_x_min, real_x_max, real_y_min, real_y_max]
    
    def get_best_next_action(self, current_coord, ground_truth):
        positive_actions = []
        negative_actions = []

        for i in range(self.n_actions):
            new_equivalent_coord = self.calculate_box_position(current_coord, i)
            if i != 0:
                reward = self.compute_reward(new_equivalent_coord, current_coord, ground_truth)
            else:
                reward = self.compute_trigger_reward(new_equivalent_coord, ground_truth)
            
            if reward>=0:
                positive_actions.append(i)
            else:
                negative_actions.append(i)
        if len(positive_actions)==0:
            return random.choice(negative_actions)
        return random.choice(positive_actions)
    
    def selection_action(self, state, current_coord, ground_truth):
        sample = random.random()
        eps_threshold = self.epsilon
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                input = Variable(state).to(self.device)

            qval = self.policy_net(input)

            _, predicted = torch.max(qval.data, 1)
            action = predicted[0]
            try:
                return action.cpu().numpy()[0]
            except:
                return action.cpu().numpy()
        else:
            return self.get_best_next_action(current_coord, ground_truth)
    
    def select_action_model(self, state):
        with torch.no_grad():
            input = Variable(state).to(self.device)

        qval = self.policy_net(input)
        _, predicted = torch.max(qval.data,1)
        action = predicted[0] # + 1
        return action

    def rewrap(self, coord):
        return min(max(0, coord), 224)
    
    def get_features(self, image):
        image = image.view(1, *image.shape).to(self.device)

        features = self.extractor(image)
        return features.data
    
    def update_history(self, action):
        action_vector = torch.zeros(self.n_actions)
        action_vector[action] = 1

        for i in range(0, 8, 1):
            self.actions_history[i][:] = self.actions_history[i + 1][:]
        self.actions_history[8][:] = action_vector[:]
        return self.actions_history
    
    def compose_state(self, image):
        features = self.get_features(image)
        features = features.view(1, -1)
        history_flatten = self.actions_history.view(1,-1).to(self.device)
        state = torch.cat((features, history_flatten), 1)
        return state
    
    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.Tensor(tuple(map(lambda s: s is not None, batch.next_state))).bool()
        next_states = [s for s in batch.next_state if s is not None]
        non_final_next_states = Variable(torch.cat(next_states))
        state_batch = Variable(torch.cat(batch.state)).to(self.device)
        action_batch = Variable(torch.LongTensor(batch.action).view(-1,1))
        reward_batch = Variable(torch.FloatTensor(batch.reward).view(-1,1))

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = Variable(torch.zeros(self.BATCH_SIZE, 1))

        non_final_next_states = non_final_next_states.to(self.device) 
        
        
        with torch.no_grad():
            d = self.target_net(non_final_next_states) 
            next_state_values[non_final_mask] = d.max(1)[0].view(-1,1)
        
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        
        loss = criterion(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, train_loader):
        xmin = 0.0
        xmax = 224.0
        ymin = 0.0
        ymax = 224.0

        for i in range(self.episodes):
            

            for key, value in train_loader.items():
                image, ground_truth = extract(key, train_loader, device=self.device)
                original_image = image.clone()
                ground_truth = ground_truth[0]

                self.actions_history = torch.zeros([9, self.n_actions])
                new_image = image
                state = self.compose_state(new_image)

                original_coordinates = [xmin, xmax, ymin, ymax]
                self.current_coord = original_coordinates
                new_equivalent_coord = original_coordinates

                done = False
                t = 0
                while not done:
                    t += 1
                    action = self.selection_action(state, self.current_coord, ground_truth)
                    if action == 0:
                        next_state = None
                        closest_gt = self.get_max_bdbox(ground_truth, self.current_coord)
                        reward = self.compute_trigger_reward(self.current_coord, closest_gt)
                        done = True
                    
                    else:
                        self.actions_history = self.update_history(action)
                        new_equivalent_coord = self.calculate_box_position(self.current_coord, action)
                        new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                        new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                        new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                        new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                        
                        new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                        try:
                            new_image = transform(new_image)
                        except ValueError:
                            break                        
                        
                        next_state = self.compose_state(new_image)
                        closest_gt = self.get_max_bdbox(ground_truth, new_equivalent_coord)
                        reward = self.compute_reward(new_equivalent_coord, self.current_coord, closest_gt)
                        self.current_coord = new_equivalent_coord

                    if t == 20:
                        done = True
                        
                    self.memory.push(state, int(action), next_state, reward)

                    state = next_state
                    image = new_image
                    
                    self.optimize()
            
            if i % self.target == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save()

            if i < 5:
                self.epsilon -=0.18

            self.save()

    def get_max_bdbox(self, ground_truth, actual_coords):
        max_iou = False
        max_gt = []
        for gt in ground_truth:
            iou = self.IOU(gt, actual_coords)
            if iou > max_iou:
                max_iou = iou
                max_gt = gt
        return max_gt


    def predict_image(self, image):
        self.policy_net.eval()
        
        original_image = image.clone()
        self.actions_history = torch.zeros((9,self.n_actions))
        state = self.compose_state(image)
        
        new_image = image
        self.current_coord = [0, 224, 0, 224]
        steps = 0
        done = False
        cross_flag = True

        while not done:
            steps += 1
            action = self.select_action_model(state)
            
            if action == 0:
                next_state = None
                new_equivalent_coord = self.current_coord
                done = True
            else:
                self.actions_history = self.update_history(action)
                new_equivalent_coord = self.calculate_box_position(self.current_coord, action)
                
                new_xmin = self.rewrap(int(new_equivalent_coord[2])-16)
                new_xmax = self.rewrap(int(new_equivalent_coord[3])+16)
                new_ymin = self.rewrap(int(new_equivalent_coord[0])-16)
                new_ymax = self.rewrap(int(new_equivalent_coord[1])+16)
                
                new_image = original_image[:, new_xmin:new_xmax, new_ymin:new_ymax]
                try:
                    new_image = transform(new_image)
                except ValueError:
                    break            
                
                next_state = self.compose_state(new_image)
                self.current_coord = new_equivalent_coord
            
            if steps == 40:
                done = True
                cross_flag = False
            
            state = next_state
            image = new_image

        print("Iteration:{} - Action:{} - Position:{}".format(steps, action, new_equivalent_coord))

        show_new_bdbox(original_image, new_equivalent_coord, count=steps)

        return new_equivalent_coord, cross_flag, steps
    
    def test(self, test_loader):
        for key, value in test_loader.items():
            image, ground_truth = extract(key, test_loader, device=self.device)
            self.predict_image(image)
            print("Ground Truth:{}".format(ground_truth))
            print("----------------------------------------------------")
        print("Test Finished")

    def predict_multiple_objects(self, image, plot=False, verbose=False):
        
        new_image = image.clone()
        all_steps = 0
        bdboxes = []   
        
        while 1:
            bdbox, cross_flag, steps = self.predict_image(new_image, plot, verbose)
            bdboxes.append(bdbox)
            
            if cross_flag:
                mask = torch.ones((224,224))
                middle_x = round((bdbox[0] + bdbox[1])/2)
                middle_y = round((bdbox[2] + bdbox[3])/2)
                length_x = round((bdbox[1] - bdbox[0])/8)
                length_y = round((bdbox[3] - bdbox[2])/8)

                mask[middle_y-length_y:middle_y+length_y,int(bdbox[0]):int(bdbox[1])] = 0
                mask[int(bdbox[2]):int(bdbox[3]),middle_x-length_x:middle_x+length_x] = 0

                new_image *= mask
                
            all_steps += steps
                
            if all_steps > 100:
                break
                    
        return bdboxes
    def evaluate(self, val_loader):
        ground_truth_boxes = []
        predicted_boxes = []
        print("Predicting boxes...")
        for key, value in val_loader.items():
            image, gt_boxes = extract(key, val_loader)
            bbox = self.predict_multiple_objects(image)
            ground_truth_boxes.append(gt_boxes)
            predicted_boxes.append(bbox)

        print("Computing recall and ap...")
        stats = eval_stats_at_threshold(predicted_boxes, ground_truth_boxes)
        print("Final result : \n"+str(stats))
        return stats

