.PHONY: all clean
all: data/nietzche/train.txt data/nietzche/test.txt
clean:
	rm -Rf data

data/nietzche/full.txt:
	mkdir -p $(@D)
	wget -O $@ https://s3.amazonaws.com/text-datasets/nietzsche.txt

data/nietzche/train.txt: data/nietzche/full.txt
	head -n $$(($$(cat $< | wc -l) * 80 / 100)) $< > $@

data/nietzche/test.txt: data/nietzche/full.txt
	tail -n +$$((($$(cat $< | wc -l) * 80 / 100) + 1)) $< > $@
