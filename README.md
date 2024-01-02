A web-based application designed to facilitate in-depth analysis of judo matches by enabling users to zoom in on specific pairs of players within video footage. 
It employs the YOLOv8 object detection model to accurately identify and localize players, and it provides a user-friendly interface for seamless video navigation
and player selection.

### Key Features:

* Player Detection: Utilizes YOLOv8 to detect and isolate judo players within video frames with a high degree of accuracy.
* Pair-Focused Bounding Boxes: Merges individual player bounding boxes into larger frames that encompass each pair of combatants, enabling focused analysis of their interactions.
* User-Initiated Zoom: Allows users to select player pairs of interest by clicking on corresponding bounding boxes, triggering a zoomed-in video view that highlights their movements and techniques.
* Web-Based Accessibility: Constructs a user-friendly interface with the Flask framework, accessible across various devices and platforms for widespread engagement.
* Front-End Integration: Employs HTML, CSS, and JavaScript to create an intuitive and visually appealing user experience.

### Built with:

* YOLOv8 Pre-trained Model
* Python
* Flask Framework
* HTML, CSS, and JavaScript

### Potential Applications:

* Coaching and Training: Supports focused analysis of player techniques and strategies, aiding coaches in providing tailored feedback and optimizing training programs.
* Match Analysis and Scouting: Facilitates detailed examination of opponent tactics and patterns, enabling athletes and coaches to develop effective competition strategies.
* Arbitration and Adjudication: Offers a tool for officials to review contested calls and assess rule adherence during matches, promoting accuracy and fairness in decision-making.
* Fan Engagement: Enhances the viewing experience for judo enthusiasts by enabling them to focus on specific players and closely observe their performance.
