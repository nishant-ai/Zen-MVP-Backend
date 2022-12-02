const axios = require("axios");
const print = console.log;

// Socket.IO
const { io } = require("../server");
let messages = "";

io.on("connect", (socket) => {
  console.log(`🔌 Client Connected: ${socket.id}`);

  // Join Room
  socket.on("join-room", (room) => {
    socket.join(room);
    socket.emit("room-joined", room);
  });

  // Message 2 Room
  socket.on("message", async (message, room) => {
    messages = messages + message + ".";
    print(messages);

    // axios ml server
    const sentiment = await axios.post("http://127.0.0.1:5000/", {
      messages: messages,
    });

    console.log(sentiment.data.sentiment);
    socket.to(room).emit("sentiment", sentiment.data.sentiment);

    socket.to(room).emit("message-received", message, socket.id);
  });

  // Leave Room
  socket.on("leave-room", (room) => {
    socket.leave(room);
    socket.emit("room-left");
  });

  socket.on("disconnect", () => {
    console.log(`❌ Client Disconnected: ${socket.id}`);
  });
});
