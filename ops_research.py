from slacker import Slacker
import json
import argparse
import os
import io
import shutil
import copy
from datetime import datetime
from pick import pick
from time import sleep
import glob
from datetime import datetime
import time
import pandas as pd
import re
import urllib.request, urllib.error, urllib.parse
#import urllib2
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
# import cv2 as cv
import sys


def find_links():
    rootDir = './public_messages_location'
    count = 0
    links = []
    prefix1 = "https://"
    prefix2 =  "http://"
    for dirName, subdirList, fileList in os.walk(rootDir):
        # print('Directory: %s' % dirName)
        for fname in fileList:
            # print('\t%s' % fname)
            # print("IS IT:")
            # print(dirName)
            with open(dirName + "/" + fname) as f:
                data = json.load(f)
                #data is type dict
                #print(data[0]['ts'])
                # print(data[0].keys())
                m_len = len(data)
                #look at entry d in collection of messages in a day. 
                for d in range(m_len):
                    timeT = data[d]['ts']
                    tmstp = int(timeT[0:timeT.find('.')])
                    tt = time.ctime(tmstp)
                    entries = str(data[d]).split()
                    collected = []
                    c_hyp = []
                    # iterate over broken  by space entries
                    for e in entries:
                        if prefix1 in e or prefix2 in e:
                            if e not in c_hyp:
                                c_hyp.append(e)
                                collected.append((e, tt))
                    links.extend(collected)
    # print(len(links))
    #print(links)
    return links
    #print(links)
            
    # for filepath in glob.iglob(PATH, recursive=True):
    #     print(filepath)
    #     for item in glob.iglob(filepath):
    #         print("----------")


# create datetime object from slack timestamp ('ts') string
def parseTimeStamp( timeStamp ):
    if '.' in timeStamp:
        t_list = timeStamp.split('.')
        if len( t_list ) != 2:
            raise ValueError( 'Invalid time stamp' )
        else:
            return datetime.utcfromtimestamp( float(t_list[0]) )

# move channel files from old directory to one with new channel name
def channelRename( oldRoomName, newRoomName ):
    # check if any files need to be moved
    if not os.path.isdir( oldRoomName ):
        return
    mkdir( newRoomName )
    for fileName in os.listdir( oldRoomName ):
        shutil.move( os.path.join( oldRoomName, fileName ), newRoomName )
    os.rmdir( oldRoomName )

def writeMessageFile( fileName, messages ):
    directory = os.path.dirname(fileName)
    # if there's no data to write to the file, return
    if not messages:
        return
    if not os.path.isdir( directory ):
        mkdir( directory )
    with open(fileName, 'w') as outFile:
        json.dump( messages, outFile, indent=4)


# parse messages by date
def parseMessages(roomDir, messages, roomType):
    nameChangeFlag = roomType + "_name"

    currentFileDate = ''
    currentMessages = []
    for message in messages:
        #first store the date of the next message
        ts = parseTimeStamp( message['ts'] )
        fileDate = '{:%Y-%m-%d}'.format(ts)

        #if it's on a different day, write out the previous day's messages
        if fileDate != currentFileDate:
            outFileName = u'{room}/{file}.json'.format( room = roomDir, file = currentFileDate )
            writeMessageFile( outFileName, currentMessages )
            currentFileDate = fileDate
            currentMessages = []

        # check if current message is a name change
        # dms won't have name change events
        if roomType != "im" and ( 'subtype' in message ) and message['subtype'] == nameChangeFlag:
            roomDir = message['name']
            oldRoomPath = message['old_name']
            newRoomPath = roomDir
            channelRename( oldRoomPath, newRoomPath )

        currentMessages.append( message )
    outFileName = u'{room}/{file}.json'.format( room = roomDir, file = currentFileDate )
    writeMessageFile( outFileName, currentMessages)

def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

# Returns the conversations to download based on the command-line arguments
def selectConversations(allConversations, commandLineArg, filter, prompt):
    global args
    if isinstance(commandLineArg, list) and len(commandLineArg) > 0:
        return filter(allConversations, commandLineArg)
    elif commandLineArg != None or not anyConversationsSpecified():
        return allConversations
    else:
        return []

# channelId is the id of the channel/group/im you want to download history for.
def getHistory(pageableObject, channelId, pageSize = 100):
    messages = []
    lastTimestamp = None

    while(True):
        response = pageableObject.history(
            channel = channelId,
            latest    = lastTimestamp,
            oldest    = 0,
            count     = pageSize
        ).body

        messages.extend(response['messages'])

        if (response['has_more'] == True):
            lastTimestamp = messages[-1]['ts'] # -1 means last element in a list
            sleep(1) # Respect the Slack API rate limit
        else:
            break

    messages.sort(key = lambda message: message['ts'])

    return messages

# Returns true if any conversations were specified on the command line
def anyConversationsSpecified():
    global args
    return args.publicChannels != None

def fetchPublicChannels(channels):
    for channel in channels:
        channelDir = channel['name'].encode('utf-8')
        print(u"Fetching history for Public Channel: {0}".format(channelDir))
        channelDir = channel['name'].encode('utf-8')
        mkdir( channelDir )
        messages = getHistory(slack.channels, channel['id'])
        parseMessages( channelDir, messages, 'channel')

# get basic info about the slack channel to ensure the authentication token works
def doTestAuth():
    testAuth = slack.auth.test().body
    teamName = testAuth['team']
    currentUser = testAuth['user']
    print(u"Successfully authenticated for team {0} and user {1} ".format(teamName, currentUser))
    return testAuth

# Since Slacker does not Cache.. populate some reused lists
def bootstrapKeyValues():
    global users, channels, groups, dms
    users = slack.users.list().body['members']
    print(u"Found {0} Users".format(len(users)))
    sleep(1)
    
    channels = slack.channels.list().body['channels']
    print(u"Found {0} Public Channels".format(len(channels)))
    sleep(1)

    groups = slack.groups.list().body['groups']
    print(u"Found {0} Private Channels or Group DMs".format(len(groups)))
    sleep(1)

    dms = slack.im.list().body['ims']
    print(u"Found {0} 1:1 DM conversations\n".format(len(dms)))
    sleep(1)

def filterConversationsByName(channelsOrGroups, channelOrGroupNames):
    return [conversation for conversation in channelsOrGroups if conversation['name'] in channelOrGroupNames]

def promptForPublicChannels(channels):
    channelNames = [channel['name'] for channel in channels]
    selectedChannels = pick(channelNames, 'Select the Public Channels you want to export:', multi_select=True)
    return [channels[index] for channelName, index in selectedChannels]


# def updateCollection():
    
def stripTags(pageContents):
    startLoc = pageContents.find("<p>")
    endLoc = pageContents.rfind("<br/>")
    pageContents = pageContents[startLoc:endLoc]
    return pageContents

def getTitles(collected_links):
    collection = []
    for i in range(len(collected_links)):
        preprocess = re.sub("\'|\,|\}|\{|\]|\[|\<|\>|\(|\)|\"", "", collected_links[i][0])
        n = len(preprocess)
        print(preprocess)
        if preprocess.find("<") != -1:
            c = preprocess.find('<')
            preprocess = preprocess[c+1:n]
        n = len(preprocess)
        if preprocess.find("http") != -1:
            c = preprocess.find('http')
            preprocess = preprocess[c:n]
        n = len(preprocess)
        if preprocess.find(">") != -1:
            c = preprocess.find('>')
            preprocess = preprocess[0:c]


        # do front and back with newline char
        n = len(preprocess)
        if preprocess.find("\\n") != -1:
            c = preprocess.find('\\n')
            if c < n//2:
                preprocess = preprocess[c+1:n]
            else:
                preprocess = preprocess[0:c]
        n = len(preprocess)
        if preprocess.find("n\\") != -1:
            c = preprocess.find('n\\')
            if c < n//2:
                preprocess = preprocess[c+1:n]
            else:
                preprocess = preprocess[0:c]
        # invalid links with no data should be removed. 
        if preprocess.find("icon") != -1:
            continue
        if preprocess.find("avatar") != -1:
            continue
        if preprocess.find("zoom") != -1:
            continue
        if preprocess.find("hangouts.google") != -1:
            continue
        
        # conjoined links need to be seperated: 
        if preprocess.find("\|http") != -1:
            position = preprocess.find("\|")
            n_len = len(preprocess)
            str1 = preprocess[0:position]
            str2 = preprocess[position + 1:n_len]
            collection.append({'links': str1, 'date': collected_links[i][1]})
            collection.append({'links': str2, 'date': collected_links[i][1]})
            continue
        # if preprocess.find('\|') != -1:
        #     preprocess = preprocess[:preprocess.find('\|')]
        pbar = preprocess.split('|')
        for p in pbar:
            if "http" in p:
                p_n = p.split('\n')
                for p2 in p_n:
                    if "http" in p2:
                        collection.append({'links': p, 'date': collected_links[i][1]})
    #make df
    df = pd.DataFrame(collection)
    print(df)
    #drop duplicates
    df.sort_values(by=['links'], inplace = True) 
    df.drop_duplicates(subset ='links', keep = 'first', inplace = True) 
    print(len(collected_links))
    # df.sort_values(by=['links'])
    df.to_csv('ethio_covid_hyperlinks.csv', index=False,header=True)  
    print(df['links'])
    index = 0
    title_col = []
    print("TEXT EXTRACTION")
    # extract content. 
    links = df['links']
    dates = df['date']
    for l in links:
        l = str(l)
        if "slack" in l:
            index += 1
            continue
        if ".png" in l:
            index += 1
            continue
        if ".pdf" in l:
            index += 1
            continue
        if ".PDF" in l:
            index += 1
            continue
        if ".jpeg" in l:
            index += 1
            continue
        if "youtube" in l:
            index += 1
            continue
        if ".jpg" in l:
            index += 1
            continue
        if "video" in l:
            index += 1
            continue
        try:
            #use slack API to sign in and see those messages:
            #use this to check the pres
            html_content = requests.get(str(l)).text
            soup = BeautifulSoup(html_content, 'lxml')
            title_text = soup.title.text
            print("Title TEXT:")
            print(title_text)
            #title_text = title_text.get('title')
            # title_text = title_text.get('title')
            #title_text = soup.findAll(['title'])[0]
            # print(type(title_text))
            # print(title_text)

            #title_text = soup.title.string
            # print("ttext")
            # print(type(title_text))
            # print(title_text)
            # start = title_text.find('>')
            # end   = title_text.find('<', 7, len(title_text))
            # print("start:")
            # print(start)
            # print("end")
            # print(end)
            # title_text = title_text[start:end]
            title_col.append({'links': l, 'title' : title_text, 'date': dates[index]})

            #  #print(soup.text.strip('\n'), file=f)
            # with open('scr2.txt', 'a') as f:
            # #     #maybe also the p tag
            #     print(l, file=f)
            #     print(soup.findAll(['title']), file=f)
            #     print("*****", file=f)
            #print(r.content)
            index += 1
            print("the link is fine:")
            print(l)
            print("---------------------")
        except:
            index += 1
            print("the link is probs messed up!:")
            print(l)
            print("------------------")
    dft = pd.DataFrame(title_col)
    dft.to_csv('ethio_covid_text.csv', index=False,header=True)  
    return dft


def cluster(df_text):
    text = df_text['title']
    # Build dictionary
    diction = []
    for t in text:
        t = str(t)
        words = t.split()
        for w in words:
            w = w.replace('\"', "").replace(",", "").replace("Log", "").replace('Log', "").replace("Django", "").replace("site", "")
            if w is 'Log':
                continue
            if len(w) > 2:
                if w not in diction:
                    diction.append(w)
    n_dict = len(diction)
    n_entries = len(text)
    # count presence of dictionary words:
    #where i is the ith link entry in df
    histograms = []
    for t in text:
        t = str(t)
        h = np.zeros(len(diction))
        words = t.split()
        print("----------------")
        for w in words:
            w = w.replace('\"', "").replace(",", "").replace("Log", "").replace('Log', "").replace("Django", "").replace("site", "")
            if len(w) > 2:
                i = diction.index(w)
                print(i)
                h[i] += 1 
        # normalize histogram for this entry:
        h_sum = np.sum(h)
        print(h_sum)
        h = np.true_divide(h, h_sum)
        histograms.append(h)
    # kmeans clustering, K is # of groups
    K = 50
    #print("HISTS:")
    #print(histograms)
    kmeans = MiniBatchKMeans(n_clusters = K).fit(histograms)
    centroids = kmeans.cluster_centers_
    return centroids, diction
    
    

if __name__ == "__main__":
    # redownload messages from slack
    #updateCollection()
    # parser = argparse.ArgumentParser(description='Export Ethio-COVID-19 Slack Channel history!')
    # parser.add_argument('--token', required=True, help="Slack API token")
    # parser.add_argument(
    #     '--publicChannels',
    #     nargs='*',
    #     default=None,
    #     metavar='CHANNEL_NAME',
    #     help="Export the given Public Channels")
    # args = parser.parse_args()
    # users = []    
    # channels = []
    # slack = Slacker(args.token)

    # #test that the token works:
    # #testAuth = doTestAuth()
    # testAuth = slack.auth.test().body
    # teamName = testAuth['team']
    # currentUser = testAuth['user']
    # print(u"Successfully authenticated for team {0} and user {1} ".format(teamName, currentUser))
    # tokenOwnerId = testAuth['user_id']

    # bootstrapKeyValues()

    # outputDirectory = "public_messages_location"
    # mkdir(outputDirectory)
    # os.chdir(outputDirectory)

    # # extract and put into folders
    # selectedChannels = selectConversations(
    #     channels,
    #     args.publicChannels,
    #     filterConversationsByName,
    #     promptForPublicChannels
    #     )
    # fetchPublicChannels(selectedChannels)
    #sleep(30)

    ############ Update messages step ends ###################
    '''
    #get links outta messages
    collected_links = find_links()
    print(collected_links)
    #get titles from links
    titles = getTitles(collected_links)
    '''
    #####Phase 3, Bag of words ML model:
    df_text = pd.read_csv("ethio_covid_text.csv") 
    np.set_printoptions(threshold=sys.maxsize)

    # print(len(df_text['title']))
    centers, diction = cluster(df_text)
    # print(centers)
    top_num = 150
    #ith element of centroid_descrip describes that centroid in 7 strings
    centroid_descrip = []
    # print("dictionary size")
    # print(len(diction))
    with open('scr2.txt', 'a') as f:
        print(centers,file=f)
    for c in centers:
        descrip = []
        # print("hist")
        # print(c)
        for num in range(top_num):
            n_best = np.argmax(c, axis=0)
            descrip.append(diction[n_best])
            c[n_best] = c[n_best] * -1
        centroid_descrip.append(descrip)
    
    print(centroid_descrip)

    #print(df_text.head(100))
    #m = df['']
    # bag of words

    #response = urllib.request.urlopen(l)
    #response = urllib.urlopen(str(l))
    # webContent = response.read()
    # print("webcontent:")
    # print(webContent[:500])
    # source_text = stripTags(webContent.decode())
    # print(source_text)



    #soup.find_all(["a", "b"])