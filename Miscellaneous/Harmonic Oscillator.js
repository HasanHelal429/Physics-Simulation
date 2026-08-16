const canvas = document.getElementById("cw");
const context = canvas.getContext("2d");
let mouseY = 0;
let mouseX = 0;
let isDragging = false;
let MODE = 'DRAG_CENTER'
//let MODE = 'MOVE_LEFT_END'

function update(x_t0, x_t1, M, K, B, L, dt) {
  return 1 / (1 / dt**2 + B/(2 * M * dt))
          * (x_t1 * (2 / dt**2 - K/M) + K * L/M + x_t0 * (B/(2 * M * dt) - 1/dt**2))
}

class SpringMass {
    constructor() {
        this.M = 1
        this.K = 1
        this.B = 0.2
        this.L = 0.25
        this.x_t0 = this.L + 0.2
        this.x_t1 = this.x_t0
        this.x_t2 = this.x_t0
        this.dt = 0.1
    }
    move(draw) {
        this.x_t2 = update(this.x_t0, this.x_t1, this.M, this.K, this.B, this.L, this.dt)
        if (draw) drawMass(this)
        this.x_t0 = structuredClone(this.x_t1);
        this.x_t1 = structuredClone(this.x_t2);
    };
}

function drawMass(s) {
    context.beginPath();
    context.fillStyle = "rgb(255, 0, 0)";
    let {x_cnv: x_cnv0, L_cnv: L_cnv0} = mass2cnv_coords(s.x_t2, s.L)
    context.fillRect(L_cnv0 + x_cnv0 - 15/2, canvas.height/2 - 15/2, 15, 15);
    context.beginPath();
    context.lineWidth = 2
    context.strokeStyle = "red";
    context.moveTo(0, canvas.height/2);
  context.lineTo(x_cnv0 + L_cnv0, canvas.height/2);
  context.stroke();
    
}

function mass2cnv_coords(x_mass, springLength) {
  return {x_cnv: canvas.width * x_mass, L_cnv: canvas.width * springLength}
}
function cnv2strng_coords(x_cnv) {
  return {x_mass: x_cnv / canvas.width - 0.25}
}

function dragString(s) {
  let {x_mass} = cnv2strng_coords(mouseX, mouseY)
  if (MODE=='MOVE_LEFT_END') {
    s.y_t1[0] = s.y_t1[1] = y_str
  }
  else if (MODE=='DRAG_CENTER') {
    s.x_t1 = s.x_t0 = s.x_t2 = x_mass
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
