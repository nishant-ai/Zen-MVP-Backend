// Modules
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const cookieParser = require("cookie-parser");
const session = require("express-session");

const { createServer } = require("http");
const { Server } = require("socket.io");

// Environment setup
require("dotenv").config();

// Server Config
const app = express();
const server = createServer(app);

app.use(
  cors({
    origin: process.env.REACT_APP_URL,
    credentials: true,
  })
);
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Socket Config
exports.io = new Server(server, {
  // To be used in socket/socket.js
  cors: {
    origin: process.env.REACT_APP_URL,
  },
});
// Socket Functionality
require("./socket/socket");

// App Listen
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
