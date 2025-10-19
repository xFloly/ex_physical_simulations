import pygame
import numpy as np
import sys

pygame.init()

class Ball:
    def __init__(self, x, y, vx, vy, radius=0.4, restitution=1, color=(255, 0, 0)):
        self.pos = np.array([x, y], dtype=float)
        self.vel = np.array([vx, vy], dtype=float)
        self.radius = radius
        self.restitution = restitution
        self.color = color

        _,_,self.start_energy = self.energy([0.0, -10.0])

    def update(self, dt, gravity, air_resistance, sim_width, sim_height):
        #simple euler
        self.vel += air_resistance * gravity * dt
        self.pos += self.vel * dt

        self.detect_collision(sim_width,sim_height)

    def detect_collision(self,sim_width,sim_height):
        if self.pos[0] - self.radius < 0:
            self.pos[0] = self.radius
            self.vel[0] = -self.vel[0] * self.restitution
        if self.pos[0] + self.radius > sim_width:
            self.pos[0] = sim_width - self.radius
            self.vel[0] = -self.vel[0] * self.restitution
        if self.pos[1] - self.radius < 0:
            self.pos[1] = self.radius
            self.vel[1] = -self.vel[1] * self.restitution

    def draw(self, screen, scale, height):
        pygame.draw.circle(
            screen,
            self.color,
            (int(self.pos[0] * scale), int(height - self.pos[1] * scale)),
            int(self.radius * scale)
        )

    # def cX(x):
    #     return int(x * c_scale)

    # def cY(y):
    #     return int(height - y * c_scale)

    def energy(self, gravity):
        kinetic = 0.5 * np.dot(self.vel, self.vel)  # m=1
        potential = -gravity[1] * self.pos[1]
        total = kinetic + potential
        return kinetic, potential, total
    
    def distance_to_click(self, mx, my, scale, height):
        ball_x = self.pos[0] * scale
        ball_y = height - self.pos[1] * scale
        return np.hypot(mx - ball_x, my - ball_y)

    def bounce(self, power=15.0, horizontal_damping=0.9):
        vx = self.vel[0] * horizontal_damping
        vy = power
        self.vel = np.array([vx, vy])

class Scene:
    def __init__(self, width=1200, height=600, sim_min_width=20.0):
        #scene
        self.panel_width = 200
        self.max_bar_height = height - 40 
    
        self.width = width
        self.height = height
        
        self.sim_pixel_width = width - self.panel_width
        self.scale = min(self.sim_pixel_width, height) / sim_min_width

        self.sim_width = self.sim_pixel_width / self.scale
        self.sim_height = height / self.scale

        #physical constants
        self.gravity = np.array([0.0, -10.0])
        self.air_resistance = 0.99
        self.time_step = 1.0 / 60.0


        self.balls = []

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Cannonball Simulation")
        self.clock = pygame.time.Clock()

    def add_ball(self, x, y, vx, vy, radius=0.4, restitution=0.8, color=(255, 0, 0)):
        self.balls.append(Ball(x, y, vx, vy, radius, restitution, color))

    def update(self):
        for ball in self.balls:
            ball.update(self.time_step, self.gravity, self.air_resistance,
                        self.sim_width, self.sim_height)
            
        #look for collision
        n = len(self.balls)
        for i in range(n):
            for j in range(i + 1, n):
                b1 = self.balls[i]
                b2 = self.balls[j]
                delta = b2.pos - b1.pos
                dist = np.linalg.norm(delta)
                if dist < b1.radius + b2.radius:
                    self.resolve_collision(b1, b2, delta, dist)

    def resolve_collision(self, b1, b2, delta, dist):
        normal = delta / dist

        # Possision Correction
        overlap = b1.radius + b2.radius - dist
        if overlap > 0:
            correction = 0.5 * overlap * normal
            b1.pos -= correction
            b2.pos += correction

        v1 = b1.vel
        v2 = b2.vel
        p1 = b1.pos
        p2 = b2.pos

        # Wektor położenia i prędkości


        # if np.dot(rel_vel, rel_pos) >= 0:
        #     return

        # m = 1
        # v1' = v1 - <v1 - v2, C1 - C2> * (C1 - C2) / ||C1 - C2||^2
        # v2' = v2 + <v1 - v2, C1 - C2> * (C1 - C2) / ||C1 - C2||^2
        rel_pos = p1 - p2
        rel_vel = v1 - v2
        rel_pos_norm_sq = np.dot(rel_pos, rel_pos)
        diff = np.dot(rel_vel, rel_pos) / rel_pos_norm_sq * (rel_pos)
        v1_new = v1 - diff
        v2_new = v2 + diff

        restitution = min(b1.restitution, b2.restitution)
        b1.vel = v1_new * restitution
        b2.vel = v2_new * restitution

        # not to boom
        if np.linalg.norm(b1.vel) < 0.01:
            b1.vel[:] = 0
        if np.linalg.norm(b2.vel) < 0.01:
            b2.vel[:] = 0
                
    def draw(self):
        self.screen.fill((255, 255, 255))
        for ball in self.balls:
            ball.draw(self.screen, self.scale, self.height)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    nearest = min(self.balls, key=lambda b: b.distance_to_click(mx, my, self.scale, self.height))
                    nearest.bounce(power=18.0)


            self.update()
            self.draw()
            self.draw_energy_panel()
            pygame.display.flip()

        pygame.quit()
        sys.exit()
    
    def draw_energy_panel(self):
        panel_x = self.width - self.panel_width
        pygame.draw.rect(self.screen, (240, 240, 240), (panel_x, 0, self.panel_width, self.height))

        max_start_energy = max(ball.start_energy for ball in self.balls)

        bar_width = (self.panel_width - 40) // len(self.balls)
        for i, ball in enumerate(self.balls):
            kinetic, potential, total = ball.energy(self.gravity)

            total_height = (total / max_start_energy) * self.max_bar_height
            kinetic_height = (kinetic / max_start_energy) * self.max_bar_height
            potential_height = (potential / max_start_energy) * self.max_bar_height
            start_line_y = self.height - (ball.start_energy / max_start_energy) * self.max_bar_height

            x = panel_x + 20 + i * bar_width

            y_bottom = self.height - kinetic_height
            pygame.draw.rect(self.screen, ball.color, (x, y_bottom, bar_width - 5, kinetic_height))
            pygame.draw.rect(self.screen, (150, 150, 150), (x, y_bottom - potential_height, bar_width - 5, potential_height))
            pygame.draw.line(self.screen, (0, 0, 0), (x, start_line_y), (x + bar_width - 5, start_line_y), 2)

            pygame.draw.rect(self.screen, (0, 0, 0), (x, self.height - total_height - 1, bar_width - 5, total_height + 1), 1)




if __name__ == "__main__":
    scene = Scene()

    scene.add_ball(1.0, 1.0, 10.0, 20.0, color=(255, 0, 0))
    scene.add_ball(3.0, 2.0, -5.0, 15.0, color=(0, 0, 255))
    scene.add_ball(5.0, 1.0, 8.0, 25.0, color=(0, 255, 0))
    scene.add_ball(7.0, 3.0, -12.0, 10.0, color=(255, 255, 0))
    scene.add_ball(9.0, 4.0, 6.0, 18.0, color=(255, 0, 255))

    scene.run()