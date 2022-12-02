const router = require("express").Router();
const bcrypt = require("bcrypt");

// Model
const User = require("../models/User/User");

// * Create User
router.post("/", async (req, res) => {
  const { firstname, lastname, userId, email, password, confirmPassword } =
    req.body;

  // password Encryption
  const salt = await bcrypt.genSalt();
  const encryptedPassword = await bcrypt.hash(req.body.password, salt);

  // Validation
  if (password !== confirmPassword) {
    return res.status(400).json({ message: "Passwords do not match" });
  }

  // Check if User Already Exists
  const userExists = await User.findOne({ userId: userId });
  if (userExists) {
    return res.status(400).json({ message: "User already exists" });
  }

  // Create New User
  const newUser = new User({
    firstname,
    lastname,
    userId,
    email,
    password: encryptedPassword,
  });

  console.log(newUser);

  // Save User
  try {
    const savedUser = await newUser.save();
    res.status(201).json(savedUser);
  } catch (err) {
    res.status(500).json({ message: err.message });
  }
});

// * Get All Users
router.get("/", async (req, res) => {
  try {
    const users = await User.find();
    res.status(200).json(users);
  } catch (err) {
    res.status(500).json(err);
  }
});

// * Get User By userId
router.get("/:id", async (req, res) => {
  try {
    const user = await User.findOne({ userId: req.params.id });
    res.status(200).json(user);
  } catch (err) {
    res.status(500).json(err);
  }
});

// * Get Current User
router.get("/current", (req, res) => {
  console.log(req.user);
  res.status(200).json(req.user);
});

// TODO: UPDATE && DELETE USER

// Export Router
module.exports = router;
