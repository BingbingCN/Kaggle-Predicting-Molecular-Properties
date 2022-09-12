#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
plt.rcParams['font.family']='serif'
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
warnings.filterwarnings(action="ignore",category=FutureWarning)


# In[2]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
sub=pd.read_csv('sample_submission.csv')
structures=pd.read_csv('structures.csv')
train.head(3)


# In[3]:


structures.head(3)


# In[4]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])

    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# In[5]:


#The above work is mainly to map features to the same file, which is more intuitive
train.head(3)


# In[6]:


#feature engineering--distance
train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values
#
train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
test['dist_z'] = (test['z_0'] - test['z_1']) ** 2


# In[7]:


train['type_0'] = train['type'].apply(lambda x: x[0])
test['type_0'] = test['type'].apply(lambda x: x[0])


# In[8]:


from tqdm import tqdm
def create_features(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')

    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']


    return df
print('Feature Engineering....')
train=create_features(train)
test=create_features(test)


# In[9]:


good_columns = [
'molecule_atom_index_0_dist_min',
'molecule_atom_index_0_dist_max',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_0_dist_mean',
'molecule_atom_index_0_dist_std',
'dist',
'molecule_atom_index_1_dist_std',
'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_mean',
'molecule_atom_index_0_dist_max_diff',
'molecule_atom_index_0_dist_max_div',
'molecule_atom_index_0_dist_std_diff',
'molecule_atom_index_0_dist_std_div',
'atom_0_couples_count',
'molecule_atom_index_0_dist_min_div',
'molecule_atom_index_1_dist_std_diff',
'molecule_atom_index_0_dist_mean_div',
'atom_1_couples_count',
'molecule_atom_index_0_dist_mean_diff',
'molecule_couples',
'atom_index_1',
'molecule_dist_mean',
'molecule_atom_index_1_dist_max_diff',
'molecule_atom_index_0_y_1_std',
'molecule_atom_index_1_dist_mean_diff',
'molecule_atom_index_1_dist_std_div',
'molecule_atom_index_1_dist_mean_div',
'molecule_atom_index_1_dist_min_diff',
'molecule_atom_index_1_dist_min_div',
'molecule_atom_index_1_dist_max_div',
'molecule_atom_index_0_z_1_std',
'y_0',
'molecule_type_dist_std_diff',
'molecule_atom_1_dist_min_diff',
'molecule_atom_index_0_x_1_std',
'molecule_dist_min',
'molecule_atom_index_0_dist_min_diff',
'molecule_atom_index_0_y_1_mean_diff',
'molecule_type_dist_min',
'molecule_atom_1_dist_min_div',
'atom_index_0',
'molecule_dist_max',
'molecule_atom_1_dist_std_diff',
'molecule_type_dist_max',
'molecule_atom_index_0_y_1_max_diff',
'molecule_type_0_dist_std_diff',
'molecule_type_dist_mean_diff',
'molecule_atom_1_dist_mean',
'molecule_atom_index_0_y_1_mean_div',
'molecule_type_dist_mean_div',
'type']


# In[10]:


from sklearn.preprocessing import LabelEncoder
for f in ['atom_1', 'type_0', 'type']:
    if f in good_columns:
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))


# In[11]:


X = train[good_columns].copy()
y = train['scalar_coupling_constant']
X_test = test[good_columns].copy()
del train, test


# In[12]:


print(X.shape,X_test.shape)
total=np.append(X.values,X_test.values,axis=0)
print(total.shape)


# In[13]:


where_nan=np.isnan(total)
total[where_nan]=0
where_inf=np.isinf(total)
total[where_inf]=0


# In[14]:


from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(-1, 1))
total= scaler1.fit_transform(total)
scaler2 = MinMaxScaler(feature_range=(-1, 1))
y= scaler2.fit_transform(y.values.reshape(-1,1))


# In[15]:


X=total[:4659076]
X_test=total[4659076:]
print(len(X),len(X_test))


# In[16]:


del total


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_valid,y_train,y_valid=train_test_split(X,y,random_state=2,shuffle=True,test_size=0.2)
print('Data preprocessing.....')


# In[19]:


from torch.utils.data import Dataset,DataLoader

class MolecularDataset(Dataset):
    def __init__(self,df_x,df_y):
        self.data=df_x
        self.label=df_y
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        data_value=torch.FloatTensor(self.data[index,:])
        label_value=torch.FloatTensor(self.label[index])
        return data_value,label_value
train_dataset=MolecularDataset(df_x=x_train,df_y=y_train)
valid_dataset=MolecularDataset(df_x=x_valid,df_y=y_valid)
BATCH_SIZE=2048
train_iterator=DataLoader(train_dataset,batch_size=BATCH_SIZE)
valid_iterator=DataLoader(valid_dataset,batch_size=BATCH_SIZE)
#test
for (data,label) in train_iterator:
    print(data)
    print(label)
    break


# In[20]:


class FullyConnectedModel(nn.Module):
    def __init__(self,InputDim=51,OutputDim=1):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(InputDim,512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.02),
            nn.Linear(512,256),
            nn.LeakyReLU(0.02),
            nn.Linear(256,128),
            nn.LeakyReLU(0.02),
            nn.Linear(128,64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.5),
            nn.Linear(64,16),
            nn.LeakyReLU(0.02),
            nn.Linear(16,1)
        )

    def forward(self,data):
        return self.model(data)

def train(model, iterator, optimizer, criterion,device='cuda'):
    epoch_loss = 0
    model=model.to(device)
    model.train()
    for batch in tqdm(iterator):
        batch[0]=batch[0].to(device)
        batch[1]=batch[1].to(device)
        criterion=criterion.to(device)
        optimizer.zero_grad()
        predictions = model(batch[0])
        loss = criterion(predictions, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator,criterion,device='cuda'):
    epoch_loss = 0
    model=model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            batch[0]=batch[0].to(device)
            batch[1]=batch[1].to(device)
            criterion=criterion.to(device)
            predictions = model(batch[0])
            loss = criterion(predictions, batch[1])
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# In[21]:

print('Build Model....')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lossfunction=nn.MSELoss(reduction='mean')
fc_model=FullyConnectedModel(InputDim=51,OutputDim=1)
optimizer=optim.AdamW(fc_model.parameters())
# print('Load pre-trained model....')
# fc_model.load_state_dict(torch.load(r'fc_model.pt'))

from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter()
print('Write log....')
# In[22]:


N_epoch=100
train_loss_list=[]
valid_loss_list=[]
for i in range(N_epoch):
    train_loss=train(model=fc_model,iterator=train_iterator,criterion=lossfunction,optimizer=optimizer,device=device)
    valid_loss=evaluate(model=fc_model,iterator=valid_iterator,criterion=lossfunction,device=device)
    print("Epoch:",(i+1))
    print("Training Loss:",train_loss,"Valid Loss:",valid_loss)
    writer.add_scalar('Training/Loss',train_loss,i)
    writer.add_scalar('Validation/Loss',valid_loss,i)
    train_loss_list.append(train_loss)
    valid_loss_list.append(valid_loss)
    if i ==0:
        best_valid_loss=valid_loss
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print('Saving Model....')
        torch.save(fc_model.state_dict(), 'fc_model.pt')


# In[26]:


# class MolecularTestDataset(Dataset):
#     def __init__(self,df_x):
#         self.data=df_x
#
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self,index):
#         data_value=torch.FloatTensor(self.data[index,:])
#         return data_value
# testdataset=MolecularTestDataset(df_x=X_test)
# testdataloader=DataLoader(testdataset,batch_size=2048)
#
#
# # In[27]:
# del train_iterator,valid_iterator
#
# print('Testing:')
# fc_model.load_state_dict(torch.load(r'fc_model.pt'))
# result=[]
# fc_model=fc_model.to('cpu')
# for data in tqdm(testdataloader):
#
#     prediction=fc_model(data)
#     result.append(prediction)
#
#
# # In[38]:
#
#
# a=np.array([])
#
#
# # In[39]:
#
#
# for i in range(len(result)):
#     a=np.append(a,result[i].reshape(-1).detach().numpy())
#
#
# # In[40]:
#
#
# result=scaler2.inverse_transform(a.reshape(-1,1))
# sub['scalar_coupling_constant']=result
# sub.to_csv('submission.csv')
# print('Save submission successfully!')


plt.figure(dpi=600)
plt.plot(range(len(train_loss_list)),train_loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Training_Loss.jpg')
print('Saving training loss.jpg')
plt.figure(dpi=600)
plt.plot(range(len(valid_loss_list)),valid_loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('valid_Loss.jpg')
print('Saving valid loss.jpg')