INPUT_DIR = ./inputs/
LOG_DIR = ./checkpoints/
RUN_DIR = ./mlruns/
MAKEDATA_DIR = ./makedata/

DATA1 = tree_gravity_shadow2contour
DATA1_DIR = $(INPUT_DIR)/$(DATA1)/
MAKEDATA1_PATH = $(MAKEDATA_DIR)/$(DATA1)/download.sh

DATA2 = fillet_shadow2contour
DATA2_DIR = $(INPUT_DIR)/$(DATA2)/
MAKEDATA2_PATH = $(MAKEDATA_DIR)/$(DATA2)/download.sh

all: $(INPUT_DIR) $(LOG_DIR) $(RUN_DIR) $(DATA1_DIR) $(DATA2_DIR)

$(INPUT_DIR):
	mkdir -p $(INPUT_DIR)

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

$(RUN_DIR):
	mkdir -p $(RUN_DIR)

$(DATA1_DIR): $(MAKEDATA1_PATH)
	sh $(MAKEDATA1_PATH)

$(DATA2_DIR): $(MAKEDATA2_PATH)
	sh $(MAKEDATA2_PATH)
