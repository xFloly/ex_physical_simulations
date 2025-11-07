"""
    python collisions_3d.py --method brute
    python collisions_3d.py --method sap
    python collisions_3d.py --method bvh
    python collisions_3d.py --method bvh --headless
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import random, math, sys, time, argparse
import numpy as np

SCREEN_W, SCREEN_H = 1000, 700
NUM_BALLS = 1000
RADIUS = 0.2
MAX_SPEED = 2.0
BOX_SIZE = 10.0
FPS = 60

class Ball:
    __slots__ = ("x","y","z","vx","vy","vz","r","colliding")
    def __init__(self, x, y, z, vx, vy, vz, r=RADIUS):
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.r = r
        self.colliding = False
    def pos(self): return (self.x, self.y, self.z)

def dist2(a: Ball, b: Ball):
    dx = a.x - b.x; dy = a.y - b.y; dz = a.z - b.z
    return dx*dx + dy*dy + dz*dz

def make_balls(n):
    balls = []
    margin = RADIUS + 0.2
    for _ in range(n):
        x = random.uniform(margin, BOX_SIZE - margin)
        y = random.uniform(margin, BOX_SIZE - margin)
        z = random.uniform(margin, BOX_SIZE - margin)
        vx = random.uniform(-MAX_SPEED, MAX_SPEED)
        vy = random.uniform(-MAX_SPEED, MAX_SPEED)
        vz = random.uniform(-MAX_SPEED, MAX_SPEED)
        balls.append(Ball(x,y,z,vx,vy,vz,RADIUS))
    return balls

#  1 zad Algorithms 

def detect_collisions_bruteforce(balls):
    n = len(balls)
    collisions = 0
    for i in range(n):
        ai = balls[i]
        for j in range(i+1, n):
            bj = balls[j]
            if dist2(ai, bj) < (ai.r + bj.r)**2:
                ai.colliding = bj.colliding = True
                collisions += 1
    return collisions

def detect_collisions_sweep_and_prune(balls):
    # sort by x-min
    intervals = [(b.x - b.r, b.x + b.r, b) for b in balls]
    intervals.sort(key=lambda x: x[0])
    active = []
    collisions = 0
    for start, end, b in intervals:
        # remove ended intervals
        active = [a for a in active if a[1] > start]
        # check with active
        for _, _, a in active:
            if abs(a.y - b.y) < a.r + b.r and abs(a.z - b.z) < a.r + b.r:
                if dist2(a, b) < (a.r + b.r)**2:
                    a.colliding = b.colliding = True
                    collisions += 1
        active.append((start, end, b))
    return collisions

#  BVH 
class BVHNode:
    def __init__(self, balls):
        self.balls = balls
        self.left = None
        self.right = None
        self.aabb_min = [min(b.x-b.r for b in balls),
                         min(b.y-b.r for b in balls),
                         min(b.z-b.r for b in balls)]
        self.aabb_max = [max(b.x+b.r for b in balls),
                         max(b.y+b.r for b in balls),
                         max(b.z+b.r for b in balls)]

def overlap(a: BVHNode, b: BVHNode):
    return not (a.aabb_max[0] < b.aabb_min[0] or a.aabb_min[0] > b.aabb_max[0] or
                a.aabb_max[1] < b.aabb_min[1] or a.aabb_min[1] > b.aabb_max[1] or
                a.aabb_max[2] < b.aabb_min[2] or a.aabb_min[2] > b.aabb_max[2])

def build_bvh(balls, depth=0, max_leaf=5):
    if len(balls) <= max_leaf:
        return BVHNode(balls)
    axis = depth % 3
    balls.sort(key=lambda b: (b.x,b.y,b.z)[axis])
    mid = len(balls)//2
    node = BVHNode(balls)
    node.left = build_bvh(balls[:mid], depth+1, max_leaf)
    node.right = build_bvh(balls[mid:], depth+1, max_leaf)
    return node

def traverse_bvh(a: BVHNode, b: BVHNode):
    if not overlap(a,b): return 0
    if not a.left and not b.left:
        count = 0
        for ba in a.balls:
            for bb in b.balls:
                if ba is not bb and dist2(ba, bb) < (ba.r+bb.r)**2:
                    ba.colliding = bb.colliding = True
                    count += 1
        return count
    count = 0
    if a.left:
        count += traverse_bvh(a.left, b)
        count += traverse_bvh(a.right, b)
    elif b.left:
        count += traverse_bvh(a, b.left)
        count += traverse_bvh(a, b.right)
    return count

def detect_collisions_bvh(balls):
    root = build_bvh(balls)
    return traverse_bvh(root, root)

#  Simulation 
def step(balls, dt, method_fn):
    for b in balls:
        b.x += b.vx * dt; b.y += b.vy * dt; b.z += b.vz * dt
        b.colliding = False
        for axis, val in enumerate((b.x,b.y,b.z)):
            if val < RADIUS or val > BOX_SIZE - RADIUS:
                if axis == 0: b.vx *= -1; b.x = max(RADIUS, min(BOX_SIZE - RADIUS, b.x))
                elif axis == 1: b.vy *= -1; b.y = max(RADIUS, min(BOX_SIZE - RADIUS, b.y))
                else: b.vz *= -1; b.z = max(RADIUS, min(BOX_SIZE - RADIUS, b.z))
    t0 = time.perf_counter()
    c = method_fn(balls)
    t1 = time.perf_counter()
    print(f"Collisions={c:5d}  |  Time per frame={ (t1-t0)*1000:.3f} ms", end="\r")
    return c

#  Rendering 
def draw_ball(b):
    glPushMatrix()
    glTranslatef(b.x - BOX_SIZE/2, b.y - BOX_SIZE/2, b.z - BOX_SIZE/2)
    color = (1.0, 0.55, 0.0) if b.colliding else (0.8, 0.1, 0.1)
    glColor3f(*color)
    quad = gluNewQuadric(); gluSphere(quad, b.r, 10, 10); gluDeleteQuadric(quad)
    glPopMatrix()

def draw_box(size):
    half = size/2
    glColor3f(0.4,0.4,0.4)
    glBegin(GL_LINES)
    for x in (-half, half):
        for y in (-half, half):
            for z in (-half, half):
                glVertex3f(x,y,-half); glVertex3f(x,y,half)
                glVertex3f(x,-half,z); glVertex3f(x,half,z)
                glVertex3f(-half,y,z); glVertex3f(half,y,z)
    glEnd()

#  Main 
def run_visual(method_fn):
    pygame.init()
    pygame.display.set_mode((SCREEN_W, SCREEN_H), DOUBLEBUF|OPENGL)
    gluPerspective(45, (SCREEN_W/SCREEN_H), 0.1, 100.0)
    glTranslatef(0,0,-18)
    glEnable(GL_DEPTH_TEST)
    balls = make_balls(NUM_BALLS)
    clock = pygame.time.Clock()
    angle = 0
    while True:
        dt = clock.tick(FPS)/1000
        for ev in pygame.event.get():
            if ev.type == QUIT: return
            if ev.type == KEYDOWN and ev.key==K_ESCAPE: return
        step(balls, dt, method_fn)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glPushMatrix(); glRotatef(15,1,0,0); glRotatef(angle,0,1,0)
        angle += 10*dt
        draw_box(BOX_SIZE)
        for b in balls: draw_ball(b)
        glPopMatrix(); pygame.display.flip()

def run_headless(method_fn, steps=200):
    balls = make_balls(NUM_BALLS)
    for s in range(steps):
        step(balls, 1/FPS, method_fn)
    print("\nDone.")

#  Entry 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["brute","sap","bvh"], default="brute")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    method_map = {
        "brute": detect_collisions_bruteforce,
        "sap": detect_collisions_sweep_and_prune,
        "bvh": detect_collisions_bvh
    }

    method_fn = method_map[args.method]
    if args.headless:
        run_headless(method_fn)
    else:
        run_visual(method_fn)
