#pragma once

struct Pos {
	int x, y;
};

class SceneData {
public:
	SceneData() :
		cameraCmd(0),
		cameraRotateX(0),
		cameraRotateY(0),
		cameraZoom(0)
	{}

	void reset() {
		cameraCmd = 0;
		cameraRotateX = 0;
		cameraRotateY = 0;
		cameraZoom = 0;
	}

	int cameraCmd;
	int cameraRotateX;
	int cameraRotateY;
	int cameraZoom;
};