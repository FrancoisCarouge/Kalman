1. How to Not Crash on Mars (0:00–2:30)
Reuse the C++Now intro.

Start with the Mars Climate Orbiter: lbf·s versus N·s, a $327M loss, and a type error the compiler never saw.
Then the turn: “Your robot’s state estimator multiplies matrices of double. What stops it from making the same mistake?”
Show the welcome screen with the compile-error joke here.

2. Kalman filtering in 90 seconds (2:30–4:30)

Predict, then update. Use the three-Gaussian picture: prior, measurement, posterior.
Anchor it in ROS 2. Filters are already everywhere in this audience’s stack: robot_localization (EKF/UKF), IMU and odometry fusion, and the trackers in Nav2 and Autoware.
Keep the math light. The point is that a filter is mostly linear algebra over mixed physical quantities.

3. Typed Linear Algebra: the problem and the idea (4:30–8:00)
Introduces the TLA project. A good place for C++Now excerpts.

The problem: in an untyped matrix, a position, a velocity and an angle are all just double. Dimensions are checked at best. Units and frames never are.
The idea: each row and column index carries a type, so every element knows its unit and quantity. Matrix operations then compose those types.
Before and after: one untyped Eigen snippet, then the TLA version.
Live moment: introduce a bug (swap two state indices, or add m/s to m) and show the compiler rejecting it.

4. The Kalman project (8:00–11:00)

A filter declaration with a typed state, plus predict and update in a few lines.
The design goals: generic, header-only, safety by construction, and no framework lock-in.
Show one real example from the repository, not a toy.

5. ROS 2: the demonstrated value proposition (11:00–14:30)
The centrepiece.

The pain they already have: nav_msgs/Odometry covariance is a float64[36], a row-major array mixing m², m·rad and rad², with no types at all. Ask the room who has indexed it wrong. Hands will go up.
The opportunity: convert messages to typed vectors and matrices once, at the node boundary. Everything inside is then checked at compile time. REP-103 (SI units) and REP-105 (frames) stop being documentation and become types.
The demo: a small ROS 2 node that fuses IMU and odometry with the typed filter. Show the bug a typed filter rejects and an untyped one silently accepts.
The value, in one slide: whole bug classes removed at compile time, zero runtime cost (proven in section 7), less test burden, and a stronger safety-argument story for ISO 26262 / IEC 61508 type work.

6. Beyond ROS 2 (14:30–16:00)

The core doesn’t depend on any framework; adapters stay thin.
Walk quickly through the ecosystems: ROS 1, Autoware, Isaac ROS, PX4/uORB, micro-ROS and embedded targets, Drake, and in-house stacks.
The message: write the estimator once, typed, and deploy it anywhere C++ runs.

7. Performance (16:00–18:30)

Typed versus untyped benchmarks: same timings.
Show a Compiler Explorer side-by-side with identical generated assembly. This is the most convincing slide for a C++ audience.
Be honest about the cost: compile time and error-message readability, plus what you’ve done about each.
Fixed-size versus dynamic sizes, and which backends you support.
Put your actual numbers here. Two charts at most.

8. A gentle call to action (18:30–20:00)

“Try it in one node.” Low commitment: wrap one estimator and keep the rest of the stack as-is.
“Tell me your use case.” You want feedback, issues and real filters to support.
Close with QR codes for both repositories and the C++Now talk on YouTube.
Last line, calling back to the opening: “Let the compiler land the robot.”

Practical notes

Slide budget: about 16–20 slides. Sections 3 and 5 carry the most weight. If time runs short, cut from section 6 first.
Demos: record both the compile error and the ROS 2 node as a fallback video. Live demos at meetups fail often.
C++Now excerpts: use them for the Mars intro and the core TLA explanation in section 3. Make sections 5 and 7 new material, so people who watched the YouTube talk still get something fresh.
Q&A bait: a backup slide on “how does this work with Eigen / existing code?” That’s the question you’ll almost certainly get.