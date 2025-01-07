import pygame
import time
import json
import torch
import math
from base_vars import *
from rules import Rules
from diffusion import Grid

def draw_dashed_circle(surface, color, center, radius, dash_length = 5,
                       gap_length = 5):
    angle = 0
    total_circumference = 2 * math.pi * radius
    dash_angle = (dash_length / total_circumference) * 360
    gap_angle = (gap_length / total_circumference) * 360
    while angle < 360:
        start_x = center[0] + radius * math.cos(math.radians(angle))
        start_y = center[1] + radius * math.sin(math.radians(angle))
        end_angle = angle + dash_angle
        end_x = center[0] + radius * math.cos(math.radians(end_angle))
        end_y = center[1] + radius * math.sin(math.radians(end_angle))
        pygame.draw.line(surface, color, (start_x, start_y), (end_x, end_y), 1)
        angle += dash_angle + gap_angle

class Button:
    def __init__(self, screen, x, y, width, height, label, font,
                 default_color = (200, 200, 200), hover_color = (170, 170, 170),
                 clicked_color = (150, 150, 150), text_color = (0, 0, 0)):
        self.screen = screen
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.font = font
        self.default_color = default_color
        self.hover_color = hover_color
        self.clicked_color = clicked_color
        self.text_color = text_color
        self.current_color = default_color
        self.clicked = False

    def draw(self):
        pygame.draw.rect(self.screen, self.current_color, self.rect)
        text_surf = self.font.render(self.label, True, self.text_color)
        text_rect = text_surf.get_rect(center = self.rect.center)
        self.screen.blit(text_surf, text_rect)

    def is_hovered(self, mouse_pos):
        return self.rect.collidepoint(mouse_pos)

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered(mouse_pos):
            self.clicked = True
            self.current_color = self.clicked_color
        elif event.type == pygame.MOUSEBUTTONUP and self.clicked:
            self.clicked = False
            self.current_color = (self.hover_color if self.is_hovered(mouse_pos)
                                  else self.default_color)
            return True
        elif event.type == pygame.MOUSEMOTION:
            if self.is_hovered(mouse_pos):
                self.current_color = self.hover_color
            else:
                self.current_color = self.default_color
        return False

class UIManager:
    def __init__(self, screen, menu_width, paused):
        self.screen = screen
        self.menu_width = menu_width
        self.font = pygame.font.Font(None, 24)

        self.save_button = Button(self.screen,
                                  screen.get_width() - menu_width + 10, 10, 160,
                                  40, "Save", self.font)
        self.play_pause_button = Button(self.screen,
                                        screen.get_width() - menu_width + 10,
                                        60, 160, 40, "Play" if paused else
                                        "Pause", self.font)
        self.sight_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 110, 160, 40,
                                          "Toggle Sight", self.font)
        self.info_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 160, 160, 40,
                                          "Toggle Info", self.font)
        self.resources_toggle_button = Button(self.screen, screen.get_width() -
                                          menu_width + 10, 210, 160, 40,
                                          "Toggle Resources", self.font)

        self.show_info = False
        self.show_sight = False
        self.show_resources = True

    def handle_event(self, event, simulation):
        if self.save_button.handle_event(event):
            simulation.save_simulation()
        if self.play_pause_button.handle_event(event):
            simulation.toggle_pause()
            self.play_pause_button.label = ("Play" if simulation.paused
                                            else "Pause")
        if self.sight_toggle_button.handle_event(event):
            self.show_sight = not self.show_sight
        if self.info_toggle_button.handle_event(event):
            self.show_info = not self.show_info
        if self.resources_toggle_button.handle_event(event):
            self.show_resources = not self.show_resources

    def draw(self, state, N, Pop, E, PopStr):
        # Draw the right menu section (white background)
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.screen.get_width() - self.menu_width, 0,
                         self.menu_width, self.screen.get_height()))

        # Draw buttons
        self.save_button.draw()
        self.play_pause_button.draw()
        self.sight_toggle_button.draw()
        self.info_toggle_button.draw()
        self.resources_toggle_button.draw()

        # Display simulation state (Period, Epoch, Age, Step)
        start_y = self.screen.get_height() // 2

        period_text = self.font.render(f"Period: {state.get('period', 0)}", True,
                                      (0, 0, 0))
        epoch_text = self.font.render(f"Epoch: {state.get('epoch', 0)}", True,
                                      (0, 0, 0))
        age_text = self.font.render(f"Age: {state.get('age', 0)} " +
                                       f"(\'{state.get('crr_age_dur', 0)})",
                                       True, (0, 0, 0))
        steps_text = self.font.render(f"Step: {state.get('step', 0)}", True,
                                      (0, 0, 0))
        N_text = self.font.render(f"N: {N}", True, (0, 0, 0))
        Pop_text = self.font.render(f"Pop.: {Pop}", True, (0, 0, 0))
        E_text = self.font.render(f"E: {int(E)}k", True, (0, 0, 0))
        PopStr_text = self.font.render(f"PopStr: {PopStr}", True, (0, 0, 0))

        self.screen.blit(period_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y))
        self.screen.blit(epoch_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 30))
        self.screen.blit(age_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 60))
        self.screen.blit(steps_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 90))
        self.screen.blit(N_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 120))
        self.screen.blit(Pop_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 150))
        self.screen.blit(E_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 180))
        self.screen.blit(PopStr_text, (self.screen.get_width() -
                         self.menu_width + 10, start_y + 210))

class Simulation:
    def __init__(self, things_object, load_file = None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(
            "Experiment 137: The Logic of Life    <>    " +
            "Eon 1: Abundance    <>    " +
            "Era 3: Nanobiota"
        )
        self.things = things_object
        self.age_start_time = time.time()

        if load_file:
            with open(load_file, 'r') as f:
                saved_data = json.load(f)
                try:
                    self.grid = Grid(saved_state = saved_data["grid"])
                except:
                    self.grid = Grid()
                self.load_state(saved_data["simulation_state"])
            self.paused = True
            self.ui_manager = UIManager(self.screen, MENU_WIDTH, self.paused)
            print(f"Simulation restored from {load_file}")
            return

        self.grid = Grid()
        self.things.add_structuralUnits(240)
        self.things.initialize_bonds()

        self.paused = False
        self.ui_manager = UIManager(self.screen, MENU_WIDTH, self.paused)
        self.step, self.age, self.epoch, self.period = 0, 0, 0, 0
        self.crr_age_dur = 0

    def update_state(self):
        self.step += 1
        if self.step == 2400:
            self.age += 1
            self.step = 0
            current_time = time.time()
            self.crr_age_dur = int(current_time - self.age_start_time)
            self.age_start_time = current_time
        if self.age == 80:
            self.epoch += 1
            self.age = 0
            self.save_simulation()
        if self.epoch == 80:
            self.period += 1
            self.epoch = 0

    def get_state(self):
        return {
            'step': self.step,
            'age': self.age,
            'epoch': self.epoch,
            'period': self.period,
            'age_start_time': self.age_start_time,
            'crr_age_dur': self.crr_age_dur
        }

    def load_state(self, state):
        self.step = state.get('step', 0)
        self.age = state.get('age', 0)
        self.epoch = state.get('epoch', 0)
        self.period = state.get('period', 0)
        self.crr_age_dur = state.get('crr_age_dur', 0)

    def run(self):
        running = True
        clock = pygame.time.Clock()

        try:
            while running:
                print("\n\n==== BEGIN STEP ====\n")
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    self.ui_manager.handle_event(event, self)

                if not self.paused:
                    self.things.final_action(self.grid)
                    Rules(self, [0, 1, 2, 3])
                    self.update_state()

                self.screen.fill(colors["0"])
                if self.ui_manager.show_resources:
                    self.grid.draw(self.screen)
                self.things.draw(self.screen, self.ui_manager.show_info,
                                 self.ui_manager.show_sight)
                self.ui_manager.draw(
                    self.get_state(),
                    self.things.N,
                    self.things.Pop,
                    self.things.E,
                    self.things.structure_mask.sum().item()
                )

                pygame.display.flip()
                # clock.tick(24)
        except:
            print("Time of the error:", self.get_state())
            self.save_simulation()
            import traceback
            traceback.print_exc()

        pygame.quit()

    def save_simulation(self):
        filename = f"simulation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        combined_state = {
            "simulation_state": self.get_state(),
            "things_state": self.things.get_state(),
            "grid": self.grid.grid.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(combined_state, f)
        print(f"Simulation saved to {filename}")

    def toggle_pause(self):
        self.paused = not self.paused
