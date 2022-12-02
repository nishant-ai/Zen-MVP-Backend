const mongoose = require("mongoose");

// Connect MongoDB at URi
mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

// Connection
const db = mongoose.connection;

// If Successful Connect!
db.on("connected", () => {
  console.log(`🏢 Connected to MongoDB at ${db.host}:${db.port}`);
});

// If Error!
db.on("error", (err) => {
  console.log(`Database error:\n${err}`);
});
