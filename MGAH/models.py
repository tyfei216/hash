import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, Dim_I, Dim_T, Dim_HID, Dim_OUT):
        super().__init__()

        self.I = nn.Sequential(
            nn.Linear(Dim_I, Dim_HID),
            nn.Tanh(),
            nn.Linear(Dim_HID, Dim_OUT), 
            nn.Sigmoid()
        )
        
        self.T = nn.Sequential(
            nn.Linear(Dim_T, Dim_HID),
            nn.Tanh(),
            nn.Linear(Dim_HID, Dim_OUT), 
            nn.Sigmoid()
        )

    def forward(self, in_I, in_T):
        I_prob = self.I(in_I)
        T_prob = self.T(in_T)
        return I_prob, T_prob

class Discriminator(nn.Module):
    def __init__(self, Dim_I, Dim_T, Dim_HID, Dim_OUT):
        super().__init__()

        self.I = nn.Sequential(
            nn.Linear(Dim_I, Dim_HID),
            nn.Tanh(),
            nn.Linear(Dim_HID, Dim_OUT), 
            nn.Sigmoid()
        )
        
        self.T = nn.Sequential(
            nn.Linear(Dim_T, Dim_HID),
            nn.Tanh(),
            nn.Linear(Dim_HID, Dim_OUT), 
            nn.Sigmoid()
        )

    def forward(self, in_I, in_T):
        I_prob = self.I(in_I)
        T_prob = self.T(in_T)
        return I_prob, T_prob

Thanks for your letter, I feel honored to be invited to the interview. I would like to pick Thursday (Feb 6)  at 12:30 noon. Both phone interview and video interview are acceptable for me. If there are any suggestions or other requirements, please let me know in advance. My user name for Skype is live:tyfei216 and my phone number is +86 18121062410. Thanks!