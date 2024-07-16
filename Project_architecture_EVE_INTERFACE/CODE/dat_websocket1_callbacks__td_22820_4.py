# me - this DAT
# dat - the WebSocket DAT
import json

def onConnect(dat):
	print('connected')
	return

# me - this DAT
# dat - the WebSocket DAT

def onDisconnect(dat):
	print('disconnected')
	return

# me - this DAT
# dat - the DAT that received a message
# rowIndex - the row number the message was placed into
# message - a unicode representation of the text
# 
# Only text frame messages will be handled in this function.

def onReceiveText(dat, rowIndex, message):
	if(message == 'ping'):
		dat.sendText('pong')
		return
	data = json.loads(message)
	keypoints = data[0]['keypoints']
	xPoses = ['x']
	yPoses = ['y']
	names = ['name']
	alpha = ['alpha']
	scores = ['score']
	for point in keypoints:
		xPoses.append(point['x'])
		yPoses.append(point['y'])
		names.append(point['name'])
		scores.append(point['score'])
		if point['score'] > op('pose_threshold')['threshold']:
			alpha.append(1)
		else:
			alpha.append(0)
	table = op('pose_table')
	table.replaceRow('x', xPoses, entireRow=True)
	table.replaceRow('y', yPoses, entireRow=True)
	table.replaceRow('alpha', alpha, entireRow=True)
	table.replaceRow('name', names, entireRow=True)
	table.replaceRow('score', scores, entireRow=True)

	
	return

# def onReceiveText(dat, rowIndex, message):
#     if message == 'ping':
#         dat.sendText('pong')
#         return
#     data = json.loads(message)
#     keypoints = data[0]['keypoints']
#     xPoses = ['x']
#     yPoses = ['y']
#     names = ['name']
#     alpha = ['alpha']
#     scores = ['score']
#     for point in keypoints:
#         if any(ignore in point['name'] for ignore in ['knee', 'ear', 'elbow']) or point['score'] <= 0.3:
#             continue
#         xPoses.append(point['x'])
#         yPoses.append(point['y'])
#         names.append(point['name'])
#         scores.append(point['score'])
#         if point['score'] > op('pose_threshold')['threshold']:
#             alpha.append(1)
#         else:
#             alpha.append(0)
#     table = op('pose_table')
#     table.replaceRow('x', xPoses, entireRow=True)
#     table.replaceRow('y', yPoses, entireRow=True)
#     table.replaceRow('alpha', alpha, entireRow=True)
#     table.replaceRow('name', names, entireRow=True)
#     table.replaceRow('score', scores, entireRow=True)

#     return


# me - this DAT
# dat - the DAT that received a message
# contents - a byte array of the message contents
# 
# Only binary frame messages will be handled in this function.

def onReceiveBinary(dat, contents):
	return

# me - this DAT
# dat - the DAT that received a message
# contents - a byte array of the message contents
# 
# Only ping messages will be handled in this function.

def onReceivePing(dat, contents):
	dat.sendPong(contents) # send a reply with same message
	print('ping', contents)
	return

# me - this DAT
# dat - the DAT that received a message
# contents - a byte array of the message content
# 
# Only pong messages will be handled in this function.

def onReceivePong(dat, contents):
	return


# me - this DAT
# dat - the DAT that received a message
# message - a unicode representation of the message
#
# Use this method to monitor the websocket status messages

def onMonitorMessage(dat, message):
	return

	