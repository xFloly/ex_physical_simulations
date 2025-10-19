import pygame
import math
from pygame.locals import *
from typing import List

class Vector2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x: float = x
        self.y: float = y

    def set(self, v: "Vector2") -> None:
        self.x = v.x
        self.y = v.y

    def clone(self) -> "Vector2":
        return Vector2(self.x, self.y)

    def add(self, v: "Vector2", s: float = 1.0) -> "Vector2":
        self.x += v.x * s
        self.y += v.y * s
        return self

    def addVectors(self, a: "Vector2", b: "Vector2") -> "Vector2":
        self.x = a.x + b.x
        self.y = a.y + b.y
        return self

    def subtract(self, v: "Vector2", s: float = 1.0) -> "Vector2":
        self.x -= v.x * s
        self.y -= v.y * s
        return self

    def subtractVectors(self, a: "Vector2", b: "Vector2") -> "Vector2":
        self.x = a.x - b.x
        self.y = a.y - b.y
        return self

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def scale(self, s: float) -> "Vector2":
        self.x *= s
        self.y *= s
        return self

    def dot(self, v: "Vector2") -> float:
        return self.x * v.x + self.y * v.y

    def perp(self) -> "Vector2":
        return Vector2(-self.y, self.x)


class PhysicsScene:
    def __init__(self):
        self.gravity: Vector2 = Vector2(3.0, -10.0)
        self.dt: float = 1.0 / 60.0
        self.numSteps: int = 100
        self.paused: bool = True
        self.wireCenter: Vector2 = Vector2()
        self.wireVelocity: Vector2 = Vector2()
        self.centerMass: float = float('inf')
        self.wireRadius: float = 0.0
        self.beads: List["Bead"] = []


physicsScene = PhysicsScene()


class Bead:
    def __init__(self, radius: float, mass: float, pos: Vector2):
        self.radius: float = radius
        self.mass: float = mass
        self.pos: Vector2 = pos.clone()
        self.prevPos: Vector2 = pos.clone()
        self.vel: Vector2 = Vector2()
        self.force: float = 0.0

    def startStep(self, dt: float, gravity: Vector2) -> None:
        self.vel.add(gravity, dt)
        self.prevPos.set(self.pos)
        self.pos.add(self.vel, dt)

    def keepOnWire(self, center: Vector2, radius: float) -> float:
        dir = Vector2()
        dir.subtractVectors(self.pos, center)
        length = dir.length()
        if length == 0.0:
            return Vector2()
        dir.scale(1.0 / length)
        lam = radius - length
        self.pos.add(dir, lam)
        return dir.scale(lam / (physicsScene.dt * physicsScene.dt)) 
        # return lam

    def endStep(self, dt: float) -> None:
        self.vel.subtractVectors(self.pos, self.prevPos)
        self.vel.scale(1.0 / dt)
        damping_factor = 1 
        self.vel.scale(damping_factor)


def handle_bead_bead_collision(b1: Bead, b2: Bead):
    restitution = 1.0
    dir = Vector2()
    dir.subtractVectors(b2.pos, b1.pos)
    d = dir.length()
    if d == 0.0 or d > b1.radius + b2.radius:
        return
    dir.scale(1.0 / d)

    # korekta pozycji
    overlap = (b1.radius + b2.radius - d) / 2.0
    b1.pos.add(dir, -overlap)
    b2.pos.add(dir, overlap)

    # kolizja sprężysta
    v1 = b1.vel.dot(dir)
    v2 = b2.vel.dot(dir)

    m1 = b1.mass
    m2 = b2.mass

    newV1 = (m1 * v1 + m2 * v2 - m2 * (v1 - v2) * restitution) / (m1 + m2)
    newV2 = (m1 * v1 + m2 * v2 - m1 * (v2 - v1) * restitution) / (m1 + m2)

    b1.vel.add(dir, newV1 - v1)
    b2.vel.add(dir, newV2 - v2)



def setupScene(width: int, height: int, simMinWidth: float, num_beads: int, bead_masses: List[float]) -> None:
    physicsScene.paused = True
    physicsScene.wireCenter.x = simMinWidth / 2.0
    physicsScene.wireCenter.y = simMinWidth / 2.0
    physicsScene.wireRadius = simMinWidth * 0.4
    physicsScene.beads.clear()

    for i in range(num_beads):
        angle = (2 * math.pi * i) / num_beads
        x = physicsScene.wireCenter.x + physicsScene.wireRadius * math.cos(angle)
        y = physicsScene.wireCenter.y + physicsScene.wireRadius * math.sin(angle)
        pos = Vector2(x, y)

        mass = bead_masses[i % len(bead_masses)]
        radius = 0.05 * (mass ** 0.5)
        physicsScene.beads.append(Bead(radius, mass, pos))

def simulate():
    if physicsScene.paused:
        return [0.0 for _ in physicsScene.beads]

    sdt = physicsScene.dt / physicsScene.numSteps
    bead_forces = [0.0 for _ in physicsScene.beads]

    total_reaction = Vector2()
    
    for _ in range(physicsScene.numSteps):
        for bead in physicsScene.beads:
            bead.startStep(sdt, physicsScene.gravity)
            
        total_reaction.x = 0.0
        total_reaction.y = 0.0

        for i, bead in enumerate(physicsScene.beads):
            # zwracaj wektor reakcji zamiast samej skali
            lam_vec = bead.keepOnWire(physicsScene.wireCenter, physicsScene.wireRadius)
            if lam_vec:
                total_reaction.add(lam_vec)
                bead_forces[i] = lam_vec.length() / (sdt * sdt)

        if physicsScene.centerMass != float('inf'):
            # F = m a => a = F/m
            acc = total_reaction.clone().scale(-1.0 / physicsScene.centerMass)
            physicsScene.wireVelocity.add(acc, sdt)
            physicsScene.wireCenter.add(physicsScene.wireVelocity, sdt)


        # for i, bead in enumerate(physicsScene.beads):
        #     lam = bead.keepOnWire(physicsScene.wireCenter, physicsScene.wireRadius)
        #     bead_forces[i] = abs(lam / (sdt * sdt))

        for bead in physicsScene.beads:
            bead.endStep(sdt)   

        for i in range(len(physicsScene.beads)):
            for j in range(i):
                handle_bead_bead_collision(physicsScene.beads[i], physicsScene.beads[j])

    return bead_forces

def draw(screen, scale):
    screen.fill((255, 255, 255))
    wireCenter = physicsScene.wireCenter
    wireRadius = physicsScene.wireRadius
    height = screen.get_height()

    def to_screen_coords(v):
        return (int(v.x * scale), int(height - v.y * scale))

    pygame.draw.circle(screen, (255, 0, 0), to_screen_coords(wireCenter), int(wireRadius * scale), 2)
    for bead in physicsScene.beads:
        pygame.draw.circle(screen, (255, 0, 0), to_screen_coords(bead.pos), int(bead.radius * scale))


def draw_button(screen, rect, text, font, bg_color, text_color):
    pygame.draw.rect(screen, bg_color, rect, border_radius=8)
    label = font.render(text, True, text_color)
    text_rect = label.get_rect(center=rect.center)
    screen.blit(label, text_rect)


def run():
    physicsScene.paused = False

def step():
    physicsScene.paused = False
    simulate()
    physicsScene.paused = True

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Symulacja koralików na pierścieniu z ruchem środka masy")
    parser.add_argument("--center-mass", type=float, default=float('inf'),
                        help="Masa środka układu (domyślnie nieskończona)")
    return parser.parse_args()

def main():
    args = parse_args()
    physicsScene.centerMass = args.center_mass  # <--- nowy atrybut

    pygame.init()
    width, height = 800, 600
    simMinWidth = 2.0
    scale = min(width, height) / simMinWidth
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Constrained Dynamics - Beads with Collision")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Verdana", 18)

    NUM_BEADS = 5
    bead_masses = [1.0, 1.0,1.0,1.0, 1.0]

    setupScene(width, height, simMinWidth, NUM_BEADS, bead_masses)

    button_width, button_height = 120, 40
    spacing = 20
    restart_rect = pygame.Rect(20, 20, button_width, button_height)
    run_rect = pygame.Rect(20 + button_width + spacing, 20, button_width, button_height)
    step_rect = pygame.Rect(20 + 2 * (button_width + spacing), 20, button_width, button_height)

    bead_forces = [0.0] * NUM_BEADS
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if restart_rect.collidepoint(mx, my):
                    setupScene(width, height, simMinWidth, NUM_BEADS, bead_masses)
                elif run_rect.collidepoint(mx, my):
                    run()
                elif step_rect.collidepoint(mx, my):
                    step()

        forces = simulate()
        if not physicsScene.paused:
            bead_forces = forces

        draw(screen, scale)
        draw_button(screen, restart_rect, "Restart", font, (100, 100, 100), (255, 255, 255))
        draw_button(screen, run_rect, "Run", font, (0, 150, 0), (255, 255, 255))
        draw_button(screen, step_rect, "Step", font, (0, 0, 150), (255, 255, 255))

        for i in range(min(5, len(bead_forces))):
            text = font.render(f"Bead {i+1}: {bead_forces[i]:.3f}", True, (0, 0, 0))
            screen.blit(text, (20, 100 + i * 25))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
