// Modules
const bcrypt = require("bcrypt");
const localStrategy = require("passport-local").Strategy;

// Model
const User = require("./models/User/User");

module.exports = function (passport) {
  passport.use(
    new localStrategy(async (userId, password, done) => {
      const user = await User.findOne({ userId: userId });

      if (err) throw err;
      if (!user) return done(null, false);

      bcrypt.compare(password, user.password, (err, result) => {
        if (err) throw err;

        if (result === true) {
          return done(null, user);
        } else {
          return done(null, false);
        }
      });
    })
  );

  passport.serializeUser((user, cb) => {
    cb(null, user.id);
  });

  passport.deserializeUser((id, cb) => {
    User.findOne({ _id: id }, (err, user) => {
      const userInformation = {
        userId: user.userId,
      };
      cb(err, userInformation);
    });
  });
};
