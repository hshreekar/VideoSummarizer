import sys
import os
sys.path.append(os.path.dirname(__file__).replace('src', '', 1))
import src.utils.outputVideo as outputVideo
import src.utils.imageHistogram as histUtility
import cv2 as cv
# import matplotlib
# from matplotlib import pyplot as plt
import math
import numpy as np
import shutil
import time
import multiprocessing

class InputVideo:

    def __init__(self,path,config,resize = False):
        self.path = path
        self.video = cv.VideoCapture(path)
        if(self.video.isOpened() == False):
            raise Exception('Error while opening video {}'.format(path))
        else:
            self.FRAME_COUNT = self.video.get(cv.CAP_PROP_FRAME_COUNT)
            self.FRAME_RATE = self.video.get(cv.CAP_PROP_FPS)
            self.FRAME_WIDTH = self.video.get(cv.CAP_PROP_FRAME_WIDTH)
            self.FRAME_HEIGHT = self.video.get(cv.CAP_PROP_FRAME_HEIGHT)
            self.frame_list = []
            self.keyframe_list = []
            self.diff_list_dict = {}  # Maps Params to diff list , calculate diff list per param only once
            self.feat_vect_dict = {}  # Maps Params to feature_vect list , calculate features only once
            self.vbow = None
            self.keras_model = None
            self.dr_model = None
            self.i2v = None
            self.summarization_data_path = self.path[0:self.path.index('.',1)]
            print("PATH->",self.path[0:self.path.index('.',1)])
            self.config = config
            self.resize = resize
            self.getFrameList()
            if os.path.exists(self.summarization_data_path):
                shutil.rmtree(self.summarization_data_path)  # Remove the old summarization data storage

    def reInit(self):
        self.video.release()
        self.video = cv.VideoCapture(self.path)

    def getVideoName(self):
        print("self.path.split('/')[-1]",self.path.split('/')[-1])
        return self.path.split('/')[-1]

    def getFrameList(self):
        if(len(self.frame_list) == 0):
            t1 = time.time()
            ret, frame = self.video.read() #Used to obtain individual frames from the original video
            print("Ret,Frame",ret,frame)
            i = 0
            while ret:
                i += 1
                self.frame_list.append(frame)
                ret, frame = self.video.read()
            print('{} frames from {} read in {} seconds'.format(len(self.frame_list),self.getVideoName(),round(time.time() - t1,2)))
        self.setNextFrameIndex(0) #When read again starts from index 0
        return self.frame_list

    def getFrameRate(self):
        return self.FRAME_RATE

    def getFrameCount(self):
        return self.FRAME_COUNT

    def getFrameWidth(self):
        return self.FRAME_WIDTH

    def getFrameHeight(self):
        return self.FRAME_HEIGHT

    def getLengthInSeconds(self):
        return int(self.FRAME_COUNT / self.FRAME_RATE)

    def getFormattedVideoLenghtInSeconds(self):
        seconds = self.getLengthInSeconds()
        minutes = int(seconds / 60)
        seconds = seconds % 60
        return str(minutes) + ':' + (str(seconds) if len(str(seconds)) == 2 else "0" + str(seconds))

    def getNthFrame(self, n):
        # sets index of next frame to n and returns it
        self.setNextFrameIndex(n)
        frame = self.__next__()  # might raise "Frame Count Exceeded"
        self.setNextFrameIndex(0)
        return frame

    def setNextFrameIndex(self, n):
        # set index of frame to be retrieved by __next__
        self.video.set(cv.CAP_PROP_POS_FRAMES, n)

    def getNextFrameIndex(self):
        return self.video.get(cv.CAP_PROP_POS_FRAMES)

    def getFrameAtSecond(self, sec):
        self.video.set(cv.CAP_PROP_POS_MSEC, sec * 1000)
        hasFrames, image = self.video.read()
        self.setNextFrameIndex(0)  # return the pointer to defualt in case we use __next__
        return hasFrames, image

    def getSampledFrameList(self, new_fps):
        print("##################")
        t1 = time.time()
        print("Sampling to {} FPS".format(new_fps))
        skip = int(self.FRAME_RATE / new_fps) # if the initial frame rate is 30, we need to reduce it to 3fps, 30/3->10 , we select every 10th frame
        print('{}/{} -> {}'.format(self.FRAME_RATE,new_fps,skip))
        frame_list = self.getFrameList()[::skip] if skip>=1 else self.getFrameList()[::1]
        if(self.resize):
            print("Resizing to {}x{}".format(self.config['width'],self.config['height']))
            frame_list = [cv.resize(i,(self.config['width'],self.config['height'])) for i in frame_list]
        print("Done in {} Sec".format(round(time.time()-t1)))
        return frame_list

    def getSampledInputVideo(self, fps):
        frame_list = self.getSampledFrameList(fps) # Contains frames sampled to 3fps and reduced dimmension
        # print(self.path)
        # print(self.path.split('.'))
        name = '.{}_{}.{}'.format(self.path.split('.')[1], fps, self.path.split('.')[2])
        print("Name of the Uniformly Sampled and resized video",name)
        width = self.config['width'] if self.resize else self.FRAME_WIDTH
        height = self.config['height'] if self.resize else self.FRAME_HEIGHT
        outputVideo.writeVideoToPath(frame_list, name, fps, width,height) #write video with 3fps to the path
        return InputVideo(name,self.config)

    def getPairwiseFrameTupleList(self, external_list=None):
        # Use unified List
        frame_list = self.getFrameList() if external_list == None else external_list
        return [(frame_list[i], frame_list[i + 1]) for i in range(len(frame_list) - 1)] #returns a list of tuples of adjacent frames [ No. of tuples = No. of Frames]

    def getAdjacentDifferenceList(self, method, params, showAndWait=False, external_list=None, getFeatures=False, loadCNNfromCache=False):
        # method - cnn , params - {model:keras}
        frame_list = self.getFrameList() if external_list == None else external_list
        pairwise_tuple_list = self.getPairwiseFrameTupleList() if external_list == None else self.getPairwiseFrameTupleList(external_list)
        # print("Pairwise_Tuple_List",pairwise_tuple_list)
        print("Length",len(pairwise_tuple_list))
        # print("F_List",frame_list)
        # Contains tuples of adjacent frames
        if '{}{}'.format(method, str(params)) in self.diff_list_dict.keys() and external_list == None:
            print("{} found in dict".format(method))
            diff_list, feat_list = self.diff_list_dict.get('{}{}'.format(method, str(params)))
            return feat_list if getFeatures else diff_list

        def getPreTrainedDiff(params):
            model = params['model']

            if (model == 'keras'):
                if self.keras_model == None and loadCNNfromCache == False:
                    from src.utils.keras_pretrained.keras_ft import KerasModel
                    self.keras_model = KerasModel()
                print("Obtaining Feature List for each frame in frame_list")
                feat_list = self.loadCNNfromCache() if loadCNNfromCache == True else [
                    self.keras_model.getFeatureVector(img) for img in frame_list]
                diff_list = [self.cossim(feat_list[i], feat_list[i + 1]) for i in range(len(feat_list) - 1)]
                print("Difference list is then computed by taking the similarity  of adjacent frames feature vectors in feat_list")
                print("DIFF_LIST",diff_list,"FEAT_LIST",feat_list)
                return diff_list, feat_list


        def getHSVDiff(params):
            difference_metric = params['difference_metric']
            feat_list = histUtility.hsvHistsForSeriesOfImages(frame_list)
            diff_list = histUtility.adjacentHistComparison(feat_list, difference_metric)
            return diff_list, feat_list

        returned_val = {
            'color': getHSVDiff,
            'cnn': getPreTrainedDiff
        }

        returned_diff_list, returned_feature_list = returned_val[method](params)

        print("Returned Diff List", (returned_diff_list))
        print("Returned Feature List", (returned_feature_list))

        if external_list == None:
            self.diff_list_dict['{}{}'.format(method, str(params))] = returned_diff_list, returned_feature_list
        print("diff_list_dict",self.diff_list_dict)

        return returned_feature_list if getFeatures else returned_diff_list

    """
    Scene Cutting Methodology
    """

    def __getCutThreshold(self, diff_list):  # we may change this to get bigger scenes
        # import statistics
        # if(self.config['scene_cut_thresh']=='auto'):
        #     discrete = list(set(int(x * 100) for x in diff_list))  # A list of the set of discretized differences
        #     discrete.sort()
        #     return statistics.median(discrete)/100,statistics.median(discrete)/100,
        #     discrete2 = discrete[:len(discrete) * 3 // 4]
        #     return 0.01 * sum(discrete) / len(discrete) if len(discrete) != 0 else 0, 0.01 * sum(discrete2) / len(discrete2) if len(discrete2) != 0 else 0
        # else:
        return self.config['scene_cut_thresh'],self.config['scene_cut_thresh']

    def getSceneBoundariesFromThreshCut(self, method, params, disolve_window_duration):
        disolve_window_frame_limit = self.getFrameRate() * disolve_window_duration
        print("Dissolve Window Frame Limit",disolve_window_frame_limit) #6
        diff_list = self.getAdjacentDifferenceList(method, params)
        thresh1, thresh2 = self.__getCutThreshold(diff_list)
        list_below = [0] + [i for (i, val) in enumerate(diff_list) if val < thresh2] + [len(diff_list) - 1] # We will get the frames indicating the scene boundaries
        print("This list comprises of all the frames of scenes whose correlation value in diff_list is less than a particular threshold ")
        print("LIST_BELOW",list_below)
        scene_pair_list = []
        for i, val in enumerate(list_below[:-1]):
            if(list_below[i + 1] - val > disolve_window_frame_limit):
                scene_pair_list.append((val, list_below[i + 1]))
        print("The below list comprises of scene boundaries obtained by comparing and checking adjacent frames with the minimum scene cut threshold ")
        print("SCENE_PAIR_LIST",scene_pair_list)
        return scene_pair_list

    def writeAndGetScenes(self, scene_boundaries_list):
        import src.utils.scene as scene
        scenes_list = []
        print("path",self.summarization_data_path)
        scenes_folder_path = self.summarization_data_path + '/scenes'
        os.makedirs(scenes_folder_path)
        kfs_per_scene_path = self.summarization_data_path + '/kfs_per_scene'
        os.makedirs(kfs_per_scene_path)

        for i, scene_boundaries in enumerate(scene_boundaries_list):
            scene_path = '{}/{}.{}'.format(scenes_folder_path, i, self.path.split('.')[2])
            scene_kf_path = '{}/{}'.format(kfs_per_scene_path,i)
            os.makedirs(scene_kf_path)
            outputVideo.writeVideoToPath(self.getFrameList(
            )[scene_boundaries[0] + 1:scene_boundaries[1] + 1], scene_path, self.getFrameRate(), self.FRAME_WIDTH, self.FRAME_HEIGHT)
            scenes_list.append(scene.Scene(scene_path, starting_index=scene_boundaries[0] + 1, ending_index=scene_boundaries[1],
                                           diff_list_dict=self.diff_list_dict, scene_id=i, keras_model=self.keras_model,
                                            dr_model=self.dr_model,scene_kf_path=scene_kf_path,config=self.config))
        print("Scenes_List",scenes_list)

        for i, scene in enumerate(scenes_list[:-1]): #Linking Scene to its next Scene
            scenes_list[i].nextScene = scenes_list[i + 1]
        return scenes_list #Returns scenes_list objects

    # def showScenes(self, scene_list):
    #     root = tk.Tk()
    #     SceneShow(root, scene_list).pack(fill="both", expand=True)
    #     root.mainloop()

    def extractKeyframesFromScene(self, scene, proc_num=None, return_dict=None):
        if(self.config['clustering']=='kmeans'):
            kfs = scene.extractKeyFramesByKMeans(method=self.config['scene_processing_features'], params=self.config['scene_processing_features_params'])
        else:
            kfs = scene.clusterSceneKMedoid(method='cnn', params={'model': 'keras'})
        if(proc_num != None):
            return_dict[proc_num] = kfs
        else:
            return kfs

    def generateKeyframes_sequential(self):
        kfs = []
        scene_list = self.writeAndGetScenes(self.getSceneBoundariesFromThreshCut(
            self.config['scene_cut_features'],self.config['scene_cut_features_params'], self.config['min_scene_length'])) #scene cut utilizes color ie correlation
        print("path",self.summarization_data_path)
        for scene in scene_list:
            kfs += self.extractKeyframesFromScene(scene)

        print("KeyFrames extracted from scene",kfs)

        before_path = self.summarization_data_path + '/kfs_before'
        os.makedirs(before_path)
        for i, kf in enumerate(kfs):
            cv.imwrite('{}/{}.jpg'.format(before_path, i), kf.image)

        if(self.config['global_removal']):
            feature_vectors = self.getAdjacentDifferenceList('cnn', self.config['cnn_params'], getFeatures=True)
            feature_vectors = [feature_vectors[kf.index] for kf in kfs]
            final_kfs = self.finalKeyframeDuplicateRemoval(kfs, self.config['global_removal_thresh'],feature_vectors)
            after_path = self.summarization_data_path + '/kfs_after'
            os.makedirs(after_path)
            for i, kf in enumerate(final_kfs):
                cv.imwrite('{}/{}.jpg'.format(after_path, i), kf.image)
            return final_kfs
        else:
            return kfs

    def finalKeyframeDuplicateRemoval(self, kfs, thresh,feature_vectors):
        # def getConcatedImage(img1, img2):
        #     numpy_horizontal = np.hstack((img1, img2))
        #     numpy_horizontal_concat = np.concatenate((img1, img2), axis=1)
        #     cv.imshow('x', numpy_horizontal_concat)
        #     cv.waitKey()
        #     cv.destroyAllWindows()

        final_set = []

        kf_with_vec = zip(kfs, feature_vectors)

        def already_exists(kf_check, final_set):
            for kf_tuple in final_set:
                diff_1_2_cnn = self.cossim(kf_check[1], kf_tuple[1])
                feat_list = histUtility.hsvHistsForSeriesOfImages([kf_check[0].image,kf_tuple[0].image])
                diff_1_2_hsv = histUtility.adjacentHistComparison(feat_list,'correlation')[0]
                if(diff_1_2_cnn > thresh or diff_1_2_hsv>self.config['global_hsv_thresh']):
                    return True
            return False
        for kf_tuple in kf_with_vec:
            if(not already_exists(kf_tuple, final_set)):
                final_set.append(kf_tuple)

        final_set = [kf[0] for kf in final_set]
        return final_set

    def cossim(self, vec1, vec2):
        from scipy import spatial
        return 0.5 * (2 - spatial.distance.cosine(vec1, vec2))  # cosine sim between a and b


