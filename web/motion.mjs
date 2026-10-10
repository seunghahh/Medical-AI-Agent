// Coordinates are in the hospital's 384 × 256 tile map.
export const homes = [
  [104, 111],
  [270, 115],
  [48, 183],
  [181, 135],
  [222, 172],
  [327, 151],
];
export const stations = {
  bed: [94, 87],
  records: [294, 112],
  review: [226, 174],
  evaluation: [327, 151],
};
export class ClinicMotion {
  constructor() {
    this.reset();
  }
  reset() {
    this.positions = homes.map((point) => [...point]);
    this.routes = homes.map(() => []);
    this.targets = homes.map((point) => [...point]);
  }
  move(role, target, instant = false) {
    if (instant) {
      this.positions[role] = [...target];
      this.routes[role] = [];
      this.targets[role] = [...target];
      return;
    }
    if (this.targets[role].every((value, i) => value === target[i])) return;
    this.targets[role] = [...target];
    // These short approaches stay on clear floor beside each workstation.
    this.routes[role] = [[...target]];
  }
  handle(event, instant = false) {
    switch (event.type) {
      case "run_start":
      case "case_start":
        this.reset();
        break;
      case "resolving":
        if (["SAY", "EXAM", "TEST"].includes(event.action?.action))
          this.move(0, stations.bed, instant);
        this.move(1, stations.records, instant);
        break;
      case "review_start":
        this.move(4, stations.review, instant);
        break;
      case "diagnosis":
        this.move(0, stations.bed, instant);
        break;
      case "evaluation_start":
        this.move(5, stations.evaluation, instant);
        break;
    }
  }
  step(seconds, reduced = false) {
    for (let role = 0; role < this.routes.length; role++) {
      const route = this.routes[role];
      if (reduced && route.length) {
        this.positions[role] = [...route.at(-1)];
        route.length = 0;
        continue;
      }
      let distance = Math.min(Math.max(seconds, 0), 0.1) * 85;
      while (route.length) {
        const pos = this.positions[role],
          target = route[0],
          length = Math.hypot(target[0] - pos[0], target[1] - pos[1]);
        if (length <= distance) {
          this.positions[role] = [...target];
          route.shift();
          distance -= length;
        } else {
          this.positions[role] = pos.map(
            (value, i) => value + ((target[i] - value) * distance) / length,
          );
          break;
        }
      }
    }
  }
}
