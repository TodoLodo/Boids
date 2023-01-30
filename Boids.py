import pygame
import numpy as np
import math
import random
from PIL import Image, ImageFilter


def test(arr):
    print("test", arr)
    return arr


# complex var hints
IntOrFloat = int or float


class Vector:
    """
    Vector class
    """
    def __init__(self, x: IntOrFloat = 0.0, y: IntOrFloat = 0.0) -> None:
        self.x = x
        self.y = y

    @property
    def magnitude(self) -> float:
        return math.sqrt(pow(self.x, 2) + pow(self.y, 2))

    @property
    def angle(self, rel='x') -> float:
        return math.degrees(math.acos(self.x/self.magnitude)) * -self.y/abs(self.y)

    @property
    def unitVector(self):
        return self.copy() / (1 if self.magnitude == 0 else self.magnitude)

    # math
    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __abs__(self):
        return self.magnitude

    def __add__(self, val):
        x = self.x + (
            val if (c1 := (t := type(val)) in [int, float]) else
            val[0] if (c2 := t in [list, tuple]) else
            val.x if (c3 := t == Vector) else 0)
        y = self.y + (val if c1 else val[1] if c2 else val.y if c3 else 0)

        return Vector(x, y)

    def __sub__(self, val):
        return self + (-val)

    def __mul__(self, val: IntOrFloat):
        return Vector(self.x * val, self.y * val)

    def __truediv__(self, val: IntOrFloat):
        return Vector(self.x / val, self.y / val) if val else Vector()

    #
    def copy(self):
        return Vector(self.x, self.y)

    #
    def __iter__(self):
        return iter((self.x, self.y))

    def __str__(self) -> str:
        return f"{self.x}_i{' + ' if (y := self.y) >= 0 else ' - '}{abs(y)}_j"

    # static methods
    @staticmethod
    def sum(vecs):
        if vecs:
            vec = vecs[0]

            for arg in vecs[1:]:
                vec += arg
        else:
            vec = None

        return vec

    @staticmethod
    def mean(vecs):
        return (Vector.sum(vecs) / len(vecs)) if vecs else Vector()

    @staticmethod
    def dot(a, b) -> IntOrFloat:
        return a.x*b.x+a.y*b.y


class Entity:
    def __init__(self, boundary: tuple = (1080, 1080), speed: IntOrFloat = 1, size: tuple = (0, 0)):
        self.__col = random.choice(['g', 'b', 'o'])
        self.__size = size
        self.__bound = boundary
        self.__pos = Vector(random.randint(0, boundary[0]), random.randint(0, boundary[1]))
        self.__vel = Vector(x := random.choice([1, -1]) * speed * random.random(),
                            random.choice([1, -1]) * math.sqrt(pow(speed, 2) - pow(x, 2)))

        self.__shortTail = (self.position, self.velocity)

        self.vDistance = np.vectorize(self.distance)
        self.vRelativeAngle = np.vectorize(self.relativeAngle)

    # properties
    @property
    def color(self):
        return self.__col

    @property
    def position(self) -> Vector:
        return self.__pos

    @property
    def x(self) -> IntOrFloat:
        return self.position.x

    @property
    def y(self) -> IntOrFloat:
        return self.position.y

    @property
    def velocity(self) -> Vector:
        return self.__vel

    @property
    def angle(self) -> IntOrFloat:
        return self.velocity.angle

    @property
    def radAngle(self) -> IntOrFloat:
        return math.radians(self.angle)

    @property
    def shortTail(self) -> tuple:
        return self.__shortTail

    # accessors
    def getDrawPosition(self, size: tuple):
        return Vector(self.x - size[0] / 2, self.y - size[1] / 2)

    def distance(self, entity):
        return abs(self.shortTail[0]-entity.position)

    def relativeAngle(self, entity):
        a = self.velocity
        b = entity.position - self.position
        angle = math.degrees(math.acos(Vector.dot(a,b)/(1 if (den:=abs(a)*abs(b)) == 0 else den)))
        #print(angle)
        return angle

    # mutators
    def setBoundary(self, boundary: tuple):
        self.__bound = boundary

    def speedAdjust(self, speed):
        self.__vel = self.__vel.unitVector * speed
        return self

    def updateShortTail(self):
        self.__shortTail = (self.position, self.velocity)

    def sac(self, entities, visionRange: IntOrFloat, avoidRange: IntOrFloat, speed: IntOrFloat, separation: bool, alignment: bool, cohesion: bool):
        neighbours = entities.entities[i:=np.where((entities.entities != self) & ((diff := self.vDistance(entities.entities)) <= visionRange) & (self.vRelativeAngle(entities.entities) <= 120))]

        zero = Vector()  # zero Vector

        v1 = Vector()  # separation
        v2 = sum((e.shortTail[1] * d / visionRange for e, d in zip(neighbours, visionRange-diff[i])), start=zero)  # alignment
        v3 = sum((e.shortTail[0] for e in neighbours), start=zero)  # cohesion

        if len(neighbours):
            closeNeighbours = neighbours[i := np.where((diff:=self.vDistance(neighbours)) <= avoidRange)]

            v1 = -sum(((e.shortTail[0] - self.position) * d/avoidRange for e, d in zip(closeNeighbours, avoidRange-diff[i])), start=zero)   # separation

        numEntities = 1 if (l:=len(neighbours)) == 0 else l

        v1 = v1 if separation else zero
        v2 = v2/numEntities if alignment else zero
        v3 = v3/numEntities if cohesion else zero

        self.__vel += v1.unitVector * speed * 0.03 + v2.unitVector * speed * 0.03 + v3.unitVector * speed * 0.03

        return self

    def move(self):
        self.__pos += self.__vel
        self.__pos.x = self.__bound[0] if self.x < 0 else 0 if self.x > self.__bound[0] else self.__pos.x
        self.__pos.y = self.__bound[1] if self.y < 0 else 0 if self.y > self.__bound[1] else self.__pos.y
        return self

    #
    def __getitem__(self, index: int):
        return tuple(self)[index]

    def __iter__(self):
        return iter((self.position, self.velocity))

    def __str__(self):
        return f"Entity({str(self.__pos)})"


class Entities:
    def __init__(self, n: int = 10, speed: IntOrFloat = 1, size: tuple = (0, 0)):
        self.entities = np.array([Entity(speed=speed, size=size) for _ in range(n)])

    # properties
    @property
    def _2D(self):
        return np.array([tuple(e) for e in self])

    # mutators
    def updateAllShortTails(self):
        [e.updateShortTail() for e in self]

    #
    def __iter__(self):
        return iter(self.entities)

    def __str__(self):
        return str([str(e) for e in self])


class Main:
    def __init__(self):
        self.targetFPS = 240
        self.entityTravelTime = 5
        self.entitySize = (50, 50)

        self.avoidRange = 40
        self.visionRange = 200
        self.visionAngle = 120
        self.visionRadRange = math.radians(self.visionAngle)

        self.textPadding = 20

        self.running = True
        self.debug = True

        pygame.init()

        # arrows
        self.arrowPil = Image.open("static/eg2.png")

        self.arrowg = self.processImage((181, 254, 217))
        self.arrowb = self.processImage((145, 149, 255))
        self.arrowo = self.processImage((255, 150, 109))

        self.entities = Entities(n=20, speed=self.entitySpeed(), size=self.entitySize)

        self.run()

    def entitySpeed(self, diagonal: IntOrFloat = 1080.00, fps: IntOrFloat = None, seconds: IntOrFloat = 10.00) -> IntOrFloat:
        fps = self.targetFPS if fps is None or fps == 0 else fps
        return diagonal / (seconds * fps)

    def processImage(self, col: tuple):
        img = self.arrowPil.copy()
        w, h = img.size

        pixels = img.load()

        for x in range(w):
            for y in range(h):
                if (px := pixels[x, y]) != (0, 0, 0, 0):
                    c = tuple(list(col) + [px[3]])
                    pixels[x, y] = tuple(list(col) + [px[3]])

        return img.rotate(-45, expand=True).resize((50, 50))

    def run(self):
        self.running = True
        drawRect = False
        pause = False
        showStats = False

        separation = False
        alignment = False
        cohesion = False

        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        font = pygame.font.SysFont(None, 32)

        clock = pygame.time.Clock()

        _screen = width, height = screen.get_size()

        diagonal = math.sqrt(pow(height, 2) + pow(width, 2))

        list(map(lambda e: e.setBoundary(_screen), self.entities))  # setting boundary

        try:
            screen.fill(color="#1E1E1E")
            while self.running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    if event.type == pygame.KEYDOWN:
                        print(f"\n\n{event.key}\n\n")
                        if event.key == 27:
                            raise SystemExit
                        if event.key == 9:
                            showStats = not showStats
                        if event.key == 49:
                            separation = not separation
                        if event.key == 50:
                            alignment = not alignment
                        if event.key == 51:
                            cohesion = not cohesion
                        if event.key == 92:
                            drawRect = not drawRect
                        if event.key == 32:
                            pause = not pause

                if not pause:
                    screen.fill(color="#1E1E1E")
                    speed = self.entitySpeed(diagonal=diagonal, fps=clock.get_fps(), seconds=self.entityTravelTime)

                    list(map(lambda e: (
                        e.sac(entities=self.entities,
                              visionRange=self.visionRange,
                              avoidRange=self.avoidRange,
                              speed=speed,
                              separation=separation,
                              alignment=alignment,
                              cohesion=cohesion).speedAdjust(speed=speed).move(),
                        tmp := getattr(self, f'arrow{e.color}').rotate(a := e.angle, expand=True).filter(ImageFilter.SMOOTH_MORE),
                        screen.blit(pim := pygame.image.fromstring(tmp.tobytes(), tmp.size, tmp.mode), pos := tuple(e.getDrawPosition(tmp.size))),
                        (pygame.draw.circle(screen, (255, 0, 0), tuple(e.position), self.avoidRange, 1),
                         pygame.draw.arc(screen, (0, 255, 0), (e.position.x-self.visionRange/2, e.position.y-self.visionRange/2, self.visionRange, self.visionRange), e.radAngle-self.visionRadRange/2, e.radAngle+self.visionRadRange/2))if drawRect else None
                    ), self.entities))

                    self.entities.updateAllShortTails()

                # texts (stats <Tab>)
                if showStats:
                    # fps
                    screen.blit(font.render(str(int(fps := clock.get_fps())), True, (254, 233, 225)), (self.textPadding, self.textPadding))
                    # separation state
                    screen.blit(r1stat := font.render(f"separation: {separation}", True, (254, 233, 225)),
                                (width-self.textPadding-r1stat.get_size()[0], self.textPadding))
                    # alignment state
                    screen.blit(r2stat := font.render(f"alignment: {alignment}", True, (254, 233, 225)),
                                (width - self.textPadding - r2stat.get_size()[0], self.textPadding*2+r1stat.get_size()[1]))
                    # cohesion state
                    screen.blit(r3stat := font.render(f"cohesion: {cohesion}", True, (254, 233, 225)),
                                (width - self.textPadding - r3stat.get_size()[0], self.textPadding*3+r1stat.get_size()[1]++r2stat.get_size()[1]))

                pygame.display.flip()
                clock.tick(self.targetFPS)
        except KeyboardInterrupt:
            print("Keyboard Interruption!")
        except SystemExit:
            pass

        pygame.quit()


if __name__ == "__main__":
    Main()
