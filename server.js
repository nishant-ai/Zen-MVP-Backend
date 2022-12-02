// Modules
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const passport = require("passport");
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
app.use(
  session({
    secret: "secretcode",
    resave: true,
    saveUninitialized: true,
  })
);
app.use(cookieParser("secretcode"));
app.use(passport.initialize());
app.use(passport.session());
require("./passportConfig")(passport);

// Socket Config
exports.io = new Server(server, {
  // To be used in socket/socket.js
  cors: {
    origin: process.env.REACT_APP_URL,
  },
});
// Socket Functionality
require("./socket/socket");

// DB Config
// require("./models/database/connection");

// Routes Config
// const userRouter = require("./routes/user");
// app.use("/user", userRouter);

// app.post("/login", (req, res, next) => {
// passport.authenticate("local", (err, user, info) => {
// console.log("🚀", err, user, info);
//
// if (err) throw err;
//
//     if (!user) res.send("No User Exists");
//     else {
//       req.logIn(user, (err) => {
//         if (err) throw err;
//         res.send("Successfully Authenticated");
//         console.log(req.user);
//       });
//     }
//   })(req, res, next);
// });
// const authRouter = require("./routes/auth");
// app.use("/auth", authRouter);

// App Listen
const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
