---
theme : "black"
transition: "slide"
highlightTheme: "monokai"
slideNumber: true
title: "Typed Kalman Filter"
---

<style>
.reveal pre { width: 100%; font-size: 0.5em; }
.reveal pre code { max-height: none; }
.reveal .columns { display: flex; gap: 16px; }
.reveal .columns > div { flex: 1; min-width: 0; }
.reveal .columns pre { font-size: 0.45em; }
</style>

<section data-background-image="welcome.png" data-background-size="contain">
<aside class="notes">
Welcome to Typed Kalman Filter. Hello, my name is François Carouge. Tonight we are going to talk about safety and developer ergonomics. We will use the C++ strong type system to make Kalman filtering easier and safer. Thank you for coming tonight and having me here at this ROS By-The-Bay meetup.
</aside>
</section>

---

<img src="mars_climate_orgiter_99.png" height="600px">

<aside class="notes">
<b>Do you know which launch this is?</b><br />
The December 1998 launch of the Mars Climate Orbiter for 327 million dollars. The program reached Mars at about 57 km of altitude instead of the planned 226 km and crashed. A ground program produced impulse in pound-force seconds, the navigation software read it as newton seconds. A factor of 4.45 that no test ever saw, because both sides were just numbers.<br />
This is not a memory safety bug. This is a specification, integration, or implemenation defect. Unit tests don't catch this. Although it would be nice to catch this class of defect before our robots arrive on another planet. Let's catch these defects at compile-time!
</aside>

---
