const passport = require("passport");
const router = require("express").Router();

// * Login User
router.post("/", (req, res, next) => {
  passport.authenticate("local", (err, user, info) => {
    console.log("👨🏻", user);
    if (err) throw err;
    if (!user) return res.status(400).json({ message: "No User Exists" });
    else {
      req.logIn(user, (err) => {
        if (err) throw err;
        res.status(200).json({ message: "Successfully Authenticated" });
      });
    }
  })(req, res, next);
});

// Export Router
module.exports = router;
