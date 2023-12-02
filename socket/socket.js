const axios = require("axios");
const print = console.log;

// Socket.IO
const { io } = require("../server");
let messages = "";

const getSentiment = async (messages) => {
  let sentiment = ["NEU", 0];

  async function query(data) {
    const response = await fetch(
      "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment",
      {
        headers: {
          Authorization: "Bearer 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",
        },
        method: "POST",
        body: JSON.stringify(data),
      }
    );
    const result = await response.json();
    return result;
  }

  const response = await query({ inputs: messages });
  if (response[0][0].label == "LABEL_2")
    sentiment = ["POS", response[0][0].score];
  else if (response[0][0].label == "LABEL_1") sentiment = ["NEU", 0];
  else if (response[0][0].label == "LABEL_0")
    sentiment = ["NEG", -1 * response[0][0].score];

  return sentiment;
};

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
    const sentiment = await getSentiment(messages);

    console.log(sentiment);
    socket.to(room).emit("sentiment", sentiment[1]);

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
