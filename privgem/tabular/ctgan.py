#!/usr/bin/env python
# -*- coding: utf-8 -*

from ctgan import CTGANSynthesizer

class ctgan(CTGANSynthesizer):
    def train(self, *args, **kwds):
        self.fit(*args, **kwds)