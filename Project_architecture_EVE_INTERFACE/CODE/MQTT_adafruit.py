# me - this DAT
# 

# Called when connection established
# dat - the OP which is cooking
def onConnect(dat):
    print('connected')
    r = op('mqttclient2')
    r.subscribe('IsadoraTeles/feeds/velostat')
    return

# Called when connection failed
# dat - the OP which is cooking
# msg - reason for failure
def onConnectFailure(dat, msg):
    print('connection adafruit failed')
    return

# Called when current connection lost
# dat - the OP which is cooking
# msg - reason for failure
def onConnectionLost(dat, msg):
    print('connection adafruit lost')
    return

# Called when server receives subscription request
# dat - the OP which is cooking
def onSubscribe(dat):
    print('subscribed to adafruit ', dat)
    return

# Called when subscription request fails.
# dat - the OP which is cooking
# msg - reason for failure
def onSubscribeFailure(dat, msg):
    print('subscribe adafruit topic fail', dat, msg)
    return

# Called when server receives unsubscription request
# dat - the OP which is cooking
def onUnsubscribe(dat):
    return

# Called when unsubscription request fails.
# dat - the OP which is cooking
# msg - reason for failure
def onUnsubscribeFailure(dat, msg):
    return

# Called when server receives publish request
# dat - the OP which is cooking
def onPublish(dat):
    return

# Called when new content received from server
# dat - the OP which is cooking
# topic - topic name of the incoming message
# payload - payload of the incoming message
# qos - qos flag for of the incoming message
# retained - retained flag of the incoming message
# dup - dup flag of the incoming message

def onMessage(dat, topic, payload, qos, retained, dup):
    table = op('tableIO')

    # Convert payload into string
    payload_str = payload.decode('utf-8')
    
    if topic =='IsadoraTeles/feeds/velostat':
        updateTable(table, 'velostat', int(payload_str))

    return

def updateTable(t, key, val):
    if t.findCell(key, cols = [0]):
        row = t.findCell(key, cols = [0]).row
        t.replaceRow(row, [key, val], entireRow = True)
        
    else:
        t.appendRow([key, val])
    return
