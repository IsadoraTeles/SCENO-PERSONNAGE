const http = require("http"); // Import HTTP module
const express = require("express"); // Import Express framework
const app = express(); // Create an Express application

app.use(express.static("public")); // Serve static files from the "public" directory
// require("dotenv").config(); // Load environment variables from .env file

const serverPort = process.env.PORT || 3000; // Set server port
const server = http.createServer(app); // Create HTTP server
const WebSocket = require("ws"); // Import WebSocket module

let keepAliveId; // ID for the keep-alive interval

const clients = {}; // Object to store connected clients
let nClients = 0; // Counter for connected clients

//const { v4: uuidv4 } = require('uuid'); // Import uuid library for unique client IDs

// Create WebSocket server, use existing HTTP server in production
const wss = process.env.NODE_ENV === "production"
  ? new WebSocket.Server({ server })
  : new WebSocket.Server({ port: 5001 });

server.listen(serverPort); // Start HTTP server
console.log(`Server started on port ${serverPort} in stage ${process.env.NODE_ENV}`); // Log server start

// WebSocket connection event handler
wss.on("connection", function (ws, req) 
{
  console.log("Connection Opened"); // Log new connection
  console.log("Client size: ", wss.clients.size); // Log current client count
  nClients += 1; // Increment client counter

  if (wss.clients.size === 1) 
    {
    console.log("first connection. starting keepalive");
    keepServerAlive(); // Start keep-alive mechanism if first client connects
  }

  const clientId = nClients; // Generate unique client ID
  ws.clientId = clientId; // Assign client ID to WebSocket connection
  clients[clientId] = ws; // Store client connection
  ws.send(JSON.stringify({ type: 'client-id', id: ws.clientId })); // Send client ID to client
  console.log(`Client connected with id ${ws.clientId}`); // Log client connection

  // Message event handler for WebSocket
  ws.on("message", (data) => 
    {
    try 
    {
      const parsedData = JSON.parse(data); // Parse incoming message

      if (parsedData.type === 'pong') 
      {
        console.log('keepAlive'); // Log keep-alive response
        return;
      }

      console.log(JSON.stringify(parsedData)); // Log incoming message
      broadcast(ws, JSON.stringify(parsedData), false); // Broadcast message to other clients
    } 
    catch (error) 
    {
      console.error('Failed to parse JSON:', error); // Log JSON parsing error
    }
  });

  // Close event handler for WebSocket
  ws.on("close", () => 
  {
    console.log("closing connection"); // Log connection closing

    const message = JSON.stringify({ type: 'clientOUT', id: ws.clientId }); // Create client out message
    broadcast(ws, message, false); // Broadcast client out message to other clients
    delete clients[ws.clientId]; // Remove client from clients object
    console.log(Object.keys(clients).length); // Log remaining clients count
    console.log(`Client disconnected with id ${ws.clientId}`); // Log client disconnection

    if (wss.clients.size === 0) 
    {
      console.log("last client disconnected, stopping keepAlive interval");
      clearInterval(keepAliveId); // Stop keep-alive interval if no clients connected
    }
  });
});

// Implement broadcast function because ws doesn't have it
const broadcast = (ws, message, includeSelf) => 
{
  wss.clients.forEach((client) => 
  {
    if (client.readyState === WebSocket.OPEN && (includeSelf || client !== ws)) 
    {
      client.send(message); // Send message to client
    }
  });
};

/**
 * Sends a ping message to all connected clients every 20 seconds
 */
const keepServerAlive = () => 
{
  keepAliveId = setInterval(() => 
  {
    wss.clients.forEach((client) => 
    {
      if (client.readyState === WebSocket.OPEN) 
      {
        client.send(JSON.stringify({ type: 'ping' })); // Send ping message
      }
    });
  }, 20000);
};

// Serve index.html file for the root route
app.get('/', function(req, res) 
{
  res.sendFile(__dirname + '/public/index.html');
});
