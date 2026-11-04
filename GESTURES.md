# Supported hand gestures

GestureGrasp understands **7 command gestures**. Hold a gesture steady for the hold time shown and it
fires once. After a command fires there is a 1-second pause before the next one (STOP never waits).
Either hand works. Gestures also work tilted or sideways, except thumbs up/down, where up vs down is
the meaning.

The source of truth is [`configs/intents.yaml`](configs/intents.yaml); this page describes it.

## Commands

| # | Gesture | How to make it | Command | Hold | When it works | What happens |
|---|---|---|---|---|---|---|
| 1 | 🤙 **Call me** | Thumb and little finger out, other three fingers folded | `WAKE` | 0.6 s | Idle, after a delivery, after a STOP | Wakes the system: it starts listening for what you want |
| 2 | 👉 **Point** | Index finger out, others folded, aimed at the object | `SELECT_TARGET` | 0.8 s | Listening, or to change a proposed object | Proposes the object you point at. Nothing moves yet |
| 3 | 👍 **Thumbs up** | Fist with the thumb pointing up | `CONFIRM` | 0.7 s | An object is proposed | The robot goes and fetches the proposed object |
| 4 | 👎 **Thumbs down** | Fist with the thumb pointing down | `REJECT` | 0.7 s | An object is proposed, or just delivered | Cancels the proposal (or sets down a wrong delivery) so you can point again |
| 5 | ✋ **Stop** | Open palm facing the camera, fingers together **or spread** | `STOP` | 0.25 s | Any time the system is awake | Emergency stop: the robot halts at once. Show *Call me* to continue |
| 6 | ✌️ **Peace** | Index and middle finger up in a V | `RETURN_HOME` | 1.0 s | Listening, object proposed, fetching, after delivery | Cancels the task and sends the robot back to its dock |
| 7 | 👌 **OK** | Thumb and index finger touching in a ring, other fingers up | `ACKNOWLEDGE` | 0.8 s | After a delivery | You have the object: the robot lets go and returns to its dock |

STOP has the shortest hold on purpose and always has priority. The spread-finger palm counts as STOP
too, because holding fingers together is hard with tremor, stiff joints or missing fingers.

## Things that are deliberately *not* commands

The model is trained to ignore these, so everyday hand movement does not move the robot:

- ✊ **Fist** and a **relaxed / resting hand**
- **Holding or grabbing things** (a cup, a remote), a **grip**
- **Other signs:** rock 🤘, three fingers, finger to lips (mute), a single little or middle finger,
  thumb-and-index pinch

Measured on people the model never saw: idle and everyday hands caused **0.04 false commands per minute**
while asleep.

## A complete fetch

```
🤙 Call me      →  "Point at the object you want."
👉 Point        →  "Thumbs up to fetch the bottle, thumbs down to choose again."
👍 Thumbs up    →  "Fetching the bottle."          (robot drives, grasps, brings it to you)
                   "Here is the bottle. Show OK when done, thumbs down if it is wrong."
👌 OK           →  robot hands it over and returns to the dock
```

At any moment: ✋ **Stop** halts the robot, ✌️ **Peace** cancels and sends it home.

The prompts in quotes appear on the web page. The Pi says them aloud with `--speak`, and so does the page with *Speak prompts here*.

## Objects you can point at

`bottle`, `cup`, `cell phone`, `remote`, `book`, `mouse`, `apple`, `banana`
(configured under `pointing.allowed_classes`; YOLO recognises them, no extra training needed).

Point with the finger visible from the side, not aimed straight at the camera. If two objects are
in the same line of sight, the nearer one is proposed, so check the prompt and answer with
thumbs up or down.

## How reliable each gesture is

People held out of training, 2-second clips with tremor-like jitter, through the real decision logic
(`reports/evaluation_report.json`):

| Gesture | Fires correctly | Typical time to fire |
|---|---|---|
| Call me | 98.8% | 0.6 s |
| Point | 95.3% | 0.8 s |
| Thumbs up | 99.0% | 0.7 s |
| Thumbs down | 99.0% | 0.7 s |
| Stop | 99.8% | 0.3 s |
| Peace | 99.8% | 1.0 s |
| OK | 99.5% | 0.8 s |

## Timeouts

If nothing happens, the system steps back on its own: listening ends after 30 s, a proposed object is
dropped after 15 s, and an unacknowledged delivery returns to idle after 30 s.

## Tips for users

- Face the camera with the hand clearly in view, about 0.5–2 m away, in reasonable light.
- Hold the gesture still for about a second; small trembling is fine.
- If a command does not fire, lower the hand, then show the gesture again.
- Resting the other hand in view is fine: the system acts on the hand making a command.
