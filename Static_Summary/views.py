from django.shortcuts import render,redirect
from .models import Document
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .forms import DocumentForm
from django.conf import settings
import sys
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

sys.path.append(os.path.dirname(__file__).replace('src', '', 1))
from   src.utils.inputVideo import InputVideo
import src.utils.imageHistogram as hist
import shutil
import time
import cv2 as cv
import time

config = {
'width':299,
'height':299,
'sampling_rate':3,
'use_cached_cnn':False,
'cnn_params':{'model':'keras'},
'use_multiprocessing':True,
'scene_cut_features':'color',
'min_scene_length':2,
'scene_cut_thresh':0.65,
'scene_cut_features_params':{'difference_metric':'correlation'},
'clustering':'kmeans',
'scene_based_removal':True,
'global_removal':False,
'scene_processing_features':'cnn',
'scene_processing_features_params':{'model':'keras'},
'global_removal_thresh':0.82,
'global_hsv_thresh':0.85,
'scene_based_removal_thresh':0.8,
'cnn_vects_path':'cached_cnn_vects_2'}


def summarize(input_video):
    t1 = time.time()
    video = InputVideo(input_video,config,resize = True) #Creates an I/P Video object
    print("video [Frame_List]",video)  # I/P video Object
    print("Processing Video: {}".format(video.getVideoName()))
    print("######################################################")
    sampled_video = video.getSampledInputVideo(config['sampling_rate'])
    print("Frame_List",sampled_video) # I/P video Object
    sampled_video.getAdjacentDifferenceList('cnn', config['cnn_params'], loadCNNfromCache=config['use_cached_cnn'])
    #Will return the difference of feature vectors of individual frames as an sampled_video object
    #if(config['use_multiprocessing']):
        #kfs = sampled_video.generateKeyframes_multiprocessing()
    #else:
    kfs = sampled_video.generateKeyframes_sequential()
    #os.remove(sampled_video.path)  # remove sampled video
    kf_path = 'kfs/' + video.getVideoName()[:video.getVideoName().find('.')]
    print("KF_PATH",kf_path)
    if(os.path.isdir(kf_path)):
        shutil.rmtree(kf_path)  # remove old results if any
    os.makedirs(kf_path)
    for i, kf in enumerate(kfs):
        cv.imwrite('{}/{}.jpg'.format(kf_path, i), kf.image)
    print("Formatted length in seconds: {}".format(video.getFormattedVideoLenghtInSeconds()))
    print("Time: {}".format(time.time() - t1))
    kfs.sort(key = lambda k: k.index)
    return [kf.image for kf in kfs],kf_path,video.getFormattedVideoLenghtInSeconds()

def main(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        print("Form",form)

        if form.is_valid():
            videoFile = form.cleaned_data['videoFile']
            print("VideoFile",videoFile)
            form.save()
            path='.'+str(settings.MEDIA_URL)+'documents/'+str(videoFile)
            shutil.copy(path,str(settings.BASE_DIR)+'/Static_Summary/')
            print("Views Path",path)
            # print(videoURL)
            if 'combinedVideo' in request.POST:
                [image,kf_path,time]=summarize(path)
            else:
                print("Missing Video Path Argument")
            # print(kf_path)
            # KF = str(settings.BASE_DIR)+str(kf_path)
            # print(KF)
            # KFS = "./media/documents/test_3"
            # KFS = D:/VideoMash-master/kfs/test str(videoFile)[:-4]+'_3'

            # tempURL=str(settings.BASE_DIR)+'/'+ kf_path.split('/')[0]
            # print("TempURL",tempURL)
            # print("Video File",str(videoFile))
            # print("BASE-DIR",str(settings.BASE_DIR)+'/'+kf_path.split('/')[0])
            # shutil.make_archive(str(videoFile)[:-4],'zip',tempURL)
            # shutil.copy(settings.BASE_DIR+'/'+ kf_path.split('/')[0]+'/'+(str(videoFile)[:-4])+'.zip',str(settings.MEDIA_ROOT))
            # downloadURL=str(settings.BASE_DIR)+'/'+kf_path.split('/')[0]+'/'+(str(videoFile)[:-4])+'.zip'
            # # downloadURL='/media/'+ kf_path.split('/')[0]+'/'+(str(videoFile)[:-4])+'.zip'
            # print(downloadURL)

            tempURL=str(settings.BASE_DIR)+'/media/documents/'+str(videoFile)[:-4]+'_3'+'/kfs_before'
            print("TempURL",tempURL)
            print("Video File",str(videoFile))
            shutil.make_archive('kfs_before','zip',tempURL)
            shutil.copy(settings.BASE_DIR+'/kfs_before.zip',str(settings.MEDIA_ROOT))
            downloadURL='/media/documents/'+'kfs_before.zip'
            downloadURL='./media/kfs_before.zip'
            print(downloadURL)
            
            
            return render(request,'dwnld.html',{'downloadURL':downloadURL})

    else:
        print("=========================================")
        form = DocumentForm()
        return render(request, 'main.html', {
            'form': form
        })
    
