const canvas = document.getElementById("cw");
const context = canvas.getContext("2d");
let mouseY = 0;
let mouseX = 0;
let isDragging = false;
let MODE = 'DRAG_CENTER'
//let MODE = 'MOVE_LEFT_END'

function update(x1_t0, x1_t1, x2_t0, x2_t1, M1, M2, K1, K2, B, L1, L2, dt) {
  return [1 / (M1 / dt**2 + B/(2 * dt))
          * (x1_t1 * (2 * M1/ dt**2 - K1 - K2) + K1 * L1 + x1_t0 * (B/(2 * dt) - M1/dt**2) + K2 * (x2_t1-L2)),
          1 / (M2 / dt**2 + B/(2 * dt))
          * (x2_t1 * (2 * M2 / dt**2 - K2) + K2 * L2 + x2_t0 * (B/(2 * dt) - M2/dt**2) + x1_t1 * K2)
  ]
}

class SpringMass {
    constructor() {
        this.M1 = 1
        this.M2 = 1
        this.K1 = 1
        this.K2 = 1
        this.B = 0.1
        this.L1 = 0.12
        this.L2 = 0.12
        this.x1_t0 = this.L1 + 0.0
        this.x1_t1 = this.x1_t0
        this.x1_t2 = this.x1_t0
        this.x2_t0 = this.L2 - 0.0 + this.x1_t0
        this.x2_t1 = this.x2_t0
        this.x2_t2 = this.x2_t0
        this.dt = 0.1
    }
    move(draw) {
        const values = update(this.x1_t0, this.x1_t1, this.x2_t0, this.x2_t1, this.M1, this.M2, this.K1, this.K2, this.B, this.L1, this.L2, this.dt)
        this.x1_t2 = values[0]
        this.x2_t2 = values[1]
        if (draw) drawMass(this)
        this.x1_t0 = structuredClone(this.x1_t1);
        this.x1_t1 = structuredClone(this.x1_t2);
        this.x2_t0 = structuredClone(this.x2_t1);
        this.x2_t1 = structuredClone(this.x2_t2);
    };
}

function drawMass(s) {
    context.beginPath();
    context.fillStyle = "rgb(255, 0, 0)";
    let {x1_cnv: x1_cnv0, x2_cnv: x2_cnv0, L1_cnv: L1_cnv0, L2_cnv: L2_cnv0} = mass2cnv_coords([s.x1_t2,s.x2_t2], [s.L1, s.L2])
    context.fillRect(L1_cnv0 + x1_cnv0 - 15/2, canvas.height/2 - 15/2, 15, 15);
    context.fillRect(L1_cnv0 + x1_cnv0 + L2_cnv0 + x2_cnv0 - 15/2, canvas.height/2 - 15/2, 15, 15);
    context.beginPath();
    context.lineWidth = 2
    context.strokeStyle = "red";
    context.moveTo(0, canvas.height/2);
  context.lineTo(L1_cnv0 + x1_cnv0 + L2_cnv0 + x2_cnv0, canvas.height/2);
  context.stroke();
    
}

function mass2cnv_coords(x_mass, springLength) {
  return {x1_cnv: canvas.width * x_mass[0], x2_cnv: canvas.width * x_mass[1],
          L1_cnv: canvas.width * springLength[0], L2_cnv: canvas.width * springLength[1]}
}
function cnv2strng_coords(x_cnv) {
  return {x_mass: x_cnv / canvas.width - 0.365}
}

function dragString(s) {
  let {x_mass} = cnv2strng_coords(mouseX, mouseY)
  if (MODE=='MOVE_LEFT_END') {
    s.y_t1[0] = s.y_t1[1] = y_str
  }
  else if (MODE=='DRAG_CENTER') {
    s.x2_t1 = s.x2_t0 = s.x2_t2 = x_mass
  }
}

addEventListener("mousemove", (e) => {  
  mouseX = e.clientX;
  mouseY = e.clientY;
},
);

addEventListener("mousedown", (e) => {
  isDragging = true;
},
);

addEventListener("mouseup", (e) => {
  isDragging = false;
},
);


addEventListener("resize", () => setSize());
function setSize() {
  canvas.height = innerHeight;
  canvas.width = innerWidth;
}

function anim() {
  requestAnimationFrame(anim);
  context.fillStyle = "rgb(45, 34, 34)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  for (let i=1; i--;) { 
    s.move(draw=(i==0))
    if (isDragging) dragString(s)
  }
}

let s = new SpringMass()
setSize();
anim();
